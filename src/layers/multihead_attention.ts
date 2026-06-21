import * as tf from '@tensorflow/tfjs';
import { type LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { type Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import { causal as generateCausalMask } from "@/masks";


export interface MultiHeadAttentionArgs extends LayerArgs {
    numHeads: number;
    embedDim: number;
    useBias?: boolean;
    dropout?: number;
    causal?: boolean;
}


export interface ScaledDotProductionAttentionKwargs {
    training?: boolean;
    dropout?: number;
    causal?: boolean;
    scaling_factor?: number;
}


/**
 * This MultiHead Attention layer implements the algorithm as described in
 * the paper "Attention is all you Need" Vaswani et al., 2017.
 * 
 * @param numHeads number of attention heads to use
 * @param embedDim the embedding size of the input (input embeddings, typically the last dimension)
 * @param causal use causal masking, default `false`
 * @param dropout use dropout during the attention calculations, default `0.0`
 * @param useBias use bias for the dense sublayers, default `true`
 * 
 * The TensorFlow version uses tf.einsum, whose gradient op has not yet been
 * implemented (https://github.com/tensorflow/tfjs/pull/4955#discussion_r619219334),
 * therefore we follow the PyTorch implementation described in:
 * https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html#multiheadattention
 * https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
 * 
 * This implementation is different from TensorFlow's whose attention weights
 * are shaped [embed dim, heads, embed dim] where as PyTorch and OpenAI's attention weights
 * are shaped [embed dim, embed dim]
 * https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/nn/modules/activation.py#L1080
 * https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L53
 * 
 * TODO: implement a fast track for self attention (query = key = value)
 * where a single dense layer combines and replaces the query, key and projection layers
 * 
 * TODO: add kDim and vDim to accept key and values whose embedding dimensions differ from query's.
 */
export class MultiHeadAttention extends tf.layers.Layer {
    static className = "MultiHeadAttention";
    protected readonly numHeads: number;
    protected readonly embedDim: number; // size of embedding dim of inputs, also per attention head
    protected readonly useBias: boolean;
    protected readonly dropout: number;
    protected readonly causal: boolean; // use causal attention to mask future words

    // projection simply means matrix multiplying query, key, and value
    // with weights to create a representation of the inputs
    protected readonly queryProjection: tf.layers.Layer;
    protected readonly keyProjection: tf.layers.Layer;
    protected readonly valueProjection: tf.layers.Layer;
    protected readonly outputProjection: tf.layers.Layer;


    constructor({ numHeads, embedDim, useBias = true, dropout = 0.0, causal = false, ...args }: MultiHeadAttentionArgs) {
        super(args);

        if (embedDim % numHeads != 0) {
            throw Error(`${this.getClassName()}::constructor ${this.name} embedDim (${embedDim}) is not divisible by numHeads (${numHeads})`);
        }

        this.numHeads = numHeads;
        this.embedDim = embedDim;
        this.useBias = useBias;
        this.dropout = dropout;
        this.causal = causal;

        if (this.dropout >= 1) {
            throw Error(`${this.getClassName()}::constructor dropout must be within [0, 1)`);
        }

        // intialize the projection weights, this should be in the
        // build() function but is done here to avoid linting complaints
        this.queryProjection = tf.layers.dense({ useBias, units: embedDim });
        this.keyProjection = tf.layers.dense({ useBias, units: embedDim });
        this.valueProjection = tf.layers.dense({ useBias, units: embedDim });
        this.outputProjection = tf.layers.dense({ useBias, units: embedDim });
    }


    /**
     * Forward propagation. Provide one input tensor or three identical tensors to self-attention.
     * @param inputs a single tensor for self-attention or an array of exactly three
     *   tensors that are either identical (self-attention) or different (cross-attention)
     * @param kwargs.packingMask a mask to prevent tokens from attending across document boundaries
     */
    override call(
        inputs: tf.Tensor | tf.Tensor[],
        kwargs: Kwargs & {
            packingMask?: tf.Tensor,
            causalMask?: tf.Tensor,
        }
    ): tf.Tensor | tf.Tensor[] {
        // validate the input tensors
        if (!Array.isArray(inputs)) {
            inputs = [inputs];
        }

        // accept only 1 input (self attention) or 3 inputs (self or cross attention)
        if (inputs.length != 1 && inputs.length != 3) {
            throw Error(`${this.getClassName()}::call ${this.name} expects exactly one or three input tensors, ${inputs.length} were provided`);
        }

        for (const input of inputs) {
            if (input.shape.length != 3) {
                throw Error(`${this.getClassName()}::call ${this.name} expected input shapes of [batch, seq, embed_dim], got ${JSON.stringify(input.shape)}`);
            }
        }

        const [query, key, value] = inputs;
        const packingMask = kwargs.packingMask ?? null;
        const causalMask = kwargs.causalMask ?? null;

        return inputs.length == 3
            // cross-attention
            ? this.forward(query!, key!, value!, packingMask, causalMask, kwargs)
            // self-attention
            : this.forward(query!, query!, query!, packingMask, causalMask, kwargs);
    }


    /**
     * Forward propagation
     */
    protected forward(
        query_input: tf.Tensor,
        key_input: tf.Tensor,
        value_input: tf.Tensor,
        packing_mask: tf.Tensor | null,
        causal_mask: tf.Tensor | null,
        kwargs: Kwargs): tf.Tensor {

        // dimensions abbreviations
        // batch = the number of sequences in the input
        // seq = the length of each sequence in the input
        // dims = the size of each token's embedding
        return tf.tidy(() => {
            const { query, key, value } = this.applyInputProjections(query_input, key_input, value_input);

            // swap the seq and heads dimensions: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
            const move_head_dim_forward = [0, 2, 1, 3];

            const {
                query_split, key_split, value_split
            } = this.splitHeads(query, key, value, move_head_dim_forward);

            // apply scaled dot production attention to get [batch, seq, numHeads, embedDim]
            const spda = MultiHeadAttention.scaledDotProductionAttention(
                query_split, key_split, value_split,
                kwargs.attentionMask ?? null, packing_mask, causal_mask,
                this.dropout, this.causal, kwargs);

            // concat heads and apply the output projection
            const output = this.outputProjection.apply(
                spda.transpose(move_head_dim_forward).reshape([query_input.shape[0], -1, this.embedDim]));

            return output as tf.Tensor;
        })
    }


    protected applyInputProjections(query_input: tf.Tensor, key_input: tf.Tensor, value_input: tf.Tensor) {
        // apply input projections, this is a batched matrix multiplication operated on the last
        // dimension of query_input and first dimension of the dense layer weights,
        // [batch, seq, dims] x [dims, dims] = [batch x seq, dims] x [dims, dims] = [batch x seq, dims] = [batch, seq, dims]
        return tf.tidy(() => {
            return {
                query: this.queryProjection.apply(query_input) as tf.Tensor,
                key: this.keyProjection.apply(key_input) as tf.Tensor,
                value: this.valueProjection.apply(value_input) as tf.Tensor
            }
        })
    }


    protected splitHeads(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, shuffle: number[]) {
        // split heads and prepare for scaled dot product attention by splitting the
        // last dimension to get the heads, bring the heads forward
        // [batch, seq, dims] -> [batch, seq, heads, dims / heads] -> [batch, heads, seq, head_dim]
        const batch_size = query.shape[0];
        const split_heads = [batch_size, -1, this.numHeads, this.embedDim / this.numHeads];

        return tf.tidy(() => {
            return {
                query_split: query.reshape(split_heads).transpose(shuffle) as tf.Tensor4D,
                key_split: key.reshape(split_heads).transpose(shuffle) as tf.Tensor4D,
                value_split: value.reshape(split_heads).transpose(shuffle) as tf.Tensor4D
            }
        })
    }


    /**
     * Applies the scaled dot-product formula: softmax(QK_t / sqrt(d_k))V,
     * formula (1) of the 2017 paper Attention Is All You Need
     * 
     * @param attentionMask a mask to prevent tokens from being 
     *   attended to (usually for padding tokens). It should have the shape
     *   [batch, head, query_sequence_len, key_sequence_len]. To use in
     *   conjunction with causal masking, the tensor should be a boolean type
     *   where false indicates a masked token.
     * @param packingMask a mask to prevent tokens from attending across document boundaries
     */
    static scaledDotProductionAttention(
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        attentionMask: tf.Tensor | null,
        packingMask: tf.Tensor | null,
        causalMask: tf.Tensor | null,
        dropout: number,
        causal: boolean,
        kwargs: ScaledDotProductionAttentionKwargs = {}
    ): tf.Tensor {
        return tf.tidy(() => {
            const { training = false, scaling_factor } = kwargs;

            key.shape.forEach((val, index) => {
                if (key.shape[index] != value.shape[index]) {
                    throw Error(`scaledDotProductionAttention: expected key and value` +
                        ` to have the same shape, got ${JSON.stringify(key.shape)} (key) and` +
                        ` ${JSON.stringify(value.shape)} (value)`);
                }
            })


            // mask's shape is [..., seq, seq] where seq is the number of words/tokens in the input,
            // not adding the batch dimension yet to lessen the calculations
            const causal_mask_shape = [
                query.shape[query.shape.length - 2],
                key.shape[key.shape.length - 2]];

            let mask = tf.zeros(causal_mask_shape);

            if (causal && causal_mask_shape[0] > 1) {
                if (attentionMask && attentionMask.dtype != "bool") {
                    throw Error(`scaledDotProductionAttention: the attention mask must be undefined or a boolean type if used with causal attention`);
                }

                // apply a causal attention mask so that tokens can only attend to preceding tokens,
                // prevents looking at head
                if (causalMask) {
                    mask = causalMask;
                } else {
                    mask = generateCausalMask(causal_mask_shape[0], causal_mask_shape[1]);
                }
            }

            if (attentionMask) {
                if (attentionMask.dtype == "bool") {
                    // convert the boolean mask to float
                    // warning: do not use 1e9, it will overflow, use something smaller like 1e7
                    mask = mask.add(attentionMask.cast("float32").sub(1).mul(1e7));
                } else {
                    // this will occur only when not using causal masking,
                    // if the attention mask is not boolean, it's assumed the masking is already calculated,
                    mask = attentionMask;
                }
            }

            // 1. matrix multiply query and transposed key
            // 2. divide by scaling factor
            // 3. apply softmax to the result
            // 4. apply attention and/or causal mask
            // 5. apply dropout
            // 6. matrix multiply softmax result with value
            let pre_softmax = query
                .matMul(key, false, true)
                .div(Math.sqrt(scaling_factor ?? key.shape[key.shape.length - 1]))
                .add(mask);

            if (packingMask) {
                // packing mask is added separately because each mask within a batch may be different,
                // so it cannot be broadcasted
                pre_softmax = pre_softmax.add(packingMask);
            }

            const spda = tf.softmax(pre_softmax);

            const spda_dropout = tf.dropout(spda, training ? dropout : 0);
            const attention = spda_dropout.matMul(value);

            return attention;
        });
    }


    override build(inputShape: tf.Shape | tf.Shape[]): void {
        let input_shape: tf.Shape[] = [];

        if (Array.isArray(inputShape) && Array.isArray(inputShape[0])) {
            input_shape = inputShape as tf.Shape[];
        } else {
            input_shape = [inputShape as tf.Shape, inputShape as tf.Shape, inputShape as tf.Shape];
        }

        if (input_shape.length != 1 && input_shape.length != 3) {
            throw Error(`${this.getClassName()}::build ${this.name} accepts either exactly one or three inputs, received ${JSON.stringify(inputShape)}`);
        }

        // initialize the sublayer weights
        this.queryProjection.build(input_shape[0]);
        this.keyProjection.build(input_shape[1]);
        this.valueProjection.build(input_shape[2]);
        this.outputProjection.build(input_shape[0]);

        // the sublayer weights need to be tracked by this layer otherwise
        // backpropagation will complain about no trainable parameters found,
        // this is an extra step that TF's Python version does not need
        this.trainableWeights = [
            ...this.queryProjection.trainableWeights,
            ...this.keyProjection.trainableWeights,
            ...this.valueProjection.trainableWeights,
            ...this.outputProjection.trainableWeights
        ];

        // rename the weights otherwise they'll take on the default naming and overlap
        // each other which breaks model loading due to duplicate weight names
        let indexing = 0;

        for (const weight of this.trainableWeights) {
            const unique_name = `${this.getClassName()}_${indexing}`;
            (weight as any).name += unique_name;
            (weight as any).originalName += unique_name;
            indexing++;
        }

        super.build(inputShape);
    }


    /**
     * MultiHead attention's output is the same shape the query's.
     */
    override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return Array.isArray(inputShape) && Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
    }


    override getConfig() {
        const base_config = super.getConfig();

        const config = {
            numHeads: this.numHeads,
            embedDim: this.embedDim,
            useBias: this.useBias,
            causal: this.causal,
            dropout: this.dropout,
            name: this.name,
        }

        Object.assign(config, base_config);

        return config;
    }
}


tf.serialization.registerClass(MultiHeadAttention);
