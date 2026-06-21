import * as tf from "@tensorflow/tfjs";
import { type Kwargs } from "@tensorflow/tfjs-layers/dist/types";
import { type ActivationIdentifier } from "@tensorflow/tfjs-layers/dist/keras_format/activation_config";

import { type MultiHeadAttentionArgs } from "../layers/multihead_attention";
import { CachedRoPEMultiHeadAttention } from "../layers/cached_rope_multihead_attention";


export interface TransformerDecoderArgs extends Omit<MultiHeadAttentionArgs, "causal"> {
    activation?: "relu" | "gelu";
    dimsFeedForward?: number;
    causal?: boolean; // use causal mask for attention on inputs
}


/**
 * This class implements the transformer decoder architecture from
 * the 2017 paper "Attention Is All You Need".
 * 
 * This decoder-only transformer layer accepts one tensor input.
 * The input tensor should have the shape
 * `[ batch, sequences, embedding dims ]`.
 * 
 * Causal masking is enabled by default for the initial attention sub-layer.
 * 
 * @param numHeads number of attention heads to use
 * @param embedDim the embedding size of the input (input embeddings, typically the last dimension)
 * @param causal use causal masking on inputs (masks future inputs to prevent looking ahead), default `true`
 * @param dropout use dropout during the attention calculations, default `0.1`
 * @param activation the activation of the intermediate feed forward layer, default `relu`
 * @param dimsFeedForward the size of the intermediate feed forward layer, default `2048`
 * @param useBias use bias for the dense sublayers and multiHead attention's dense sublayers, default `true`
 */
export class TransformerDecoder extends tf.layers.Layer {
    static className = "TransformerDecoder";

    protected readonly causalSelfAttention: tf.layers.Layer;
    protected readonly causalSelfAttentionDropout: tf.layers.Layer;
    protected readonly causalSelfAttentionNorm: tf.layers.Layer;

    protected readonly feedforward1: tf.layers.Layer;
    protected readonly feedforward2: tf.layers.Layer;
    protected readonly feedForwardDropout: tf.layers.Layer;
    protected readonly feedFowardNorm: tf.layers.Layer;

    protected readonly numHeads: number;
    protected readonly embedDim: number;
    protected readonly useBias: boolean;
    protected readonly dropout: number;
    protected readonly activation: ActivationIdentifier;
    protected readonly dimsFeedForward: number;

    constructor({ numHeads, embedDim, useBias, dropout, activation, dimsFeedForward, ...args }: TransformerDecoderArgs) {
        super(args);

        this.numHeads = numHeads;
        this.embedDim = embedDim;
        this.useBias = useBias ?? true;
        this.dropout = dropout ?? 0.1;
        this.activation = activation ?? "relu";

        if (this.dropout >= 1) {
            throw Error(`${this.getClassName()}::constructor dropout must be within [0, 1)`);
        }

        // in the paper section 3.3, d_model=512 (embedDim) and first dense layer outputs d_ff=2048
        this.dimsFeedForward = dimsFeedForward ?? embedDim * 4;

        // self attention sub-block
        this.causalSelfAttention = new CachedRoPEMultiHeadAttention({
            numHeads: this.numHeads, embedDim: this.embedDim,
            useBias: this.useBias, dropout: this.dropout,
            causal: true
        });
        this.causalSelfAttentionDropout = tf.layers.dropout({ rate: this.dropout })
        this.causalSelfAttentionNorm = tf.layers.layerNormalization({ epsilon: 1e-6 });

        // feed forward sub-block
        this.feedforward1 = tf.layers.dense({
            units: this.dimsFeedForward,
            activation: this.activation,
            useBias: this.useBias,
        });
        this.feedforward2 = tf.layers.dense({
            units: this.embedDim,
            activation: "linear",
            useBias: this.useBias
        });
        this.feedForwardDropout = tf.layers.dropout({ rate: this.dropout });
        this.feedFowardNorm = tf.layers.layerNormalization({ epsilon: 1e-6 });
    }


    /**
     * Forward propagation
     * 
     * @param inputs input tensor
     * @return the output tensor
     */
    override call(inputs: tf.Tensor | tf.Tensor[], kwargs: Kwargs): tf.Tensor | tf.Tensor[] {
        // validate the input tensors
        if (Array.isArray(inputs) && inputs.length != 1 && inputs.length != 2) {
            throw Error(`${this.getClassName()}::call ${this.name} expects one input tensor, got ${inputs.length} inputs.`);
        }

        if (Array.isArray(inputs)) {
            inputs = inputs[0] as tf.Tensor;
        }

        // perform forward propagation
        return tf.tidy(() => {
            let output = this.causalSelfAttentionBlock(inputs, kwargs);
            output = this.feedForwardBlock(output, kwargs);

            return output;
        });
    }


    protected causalSelfAttentionBlock(x: tf.Tensor, kwargs: Kwargs): tf.Tensor {
        return tf.tidy(() => {
            const residual = x;

            let attention = this.causalSelfAttention.apply(x, kwargs) as tf.Tensor;
            attention = this.causalSelfAttentionDropout.apply(attention, kwargs) as tf.Tensor;
            attention = tf.add(attention, residual);
            attention = this.causalSelfAttentionNorm.apply(attention, kwargs) as tf.Tensor;

            return attention;
        });
    }


    protected feedForwardBlock(x: tf.Tensor, kwargs: Kwargs): tf.Tensor {
        return tf.tidy(() => {
            const residual = x;

            let feedForward = this.feedforward1.apply(x, kwargs);
            feedForward = this.feedforward2.apply(feedForward, kwargs);
            feedForward = this.feedForwardDropout.apply(feedForward, kwargs) as tf.Tensor;
            feedForward = tf.add(feedForward, residual);
            feedForward = this.feedFowardNorm.apply(feedForward, kwargs) as tf.Tensor;

            return feedForward;
        });
    }


    /**
     * Initialize the sublayers' weights and track them to enable serialization
     */
    override build(inputShape: tf.Shape | tf.Shape[]): void {
        let input_shapes: tf.Shape[] = [];

        if (Array.isArray(inputShape) && Array.isArray(inputShape[0])) {
            // input is an array of shapes
            input_shapes = inputShape as tf.Shape[];
        } else if (inputShape.length != 0) {
            // input is a single shape
            input_shapes = [inputShape as tf.Shape];
        }

        if (input_shapes.length != 1 && input_shapes.length != 2) {
            throw Error(`${this.getClassName()}::build ${this.name} expects an input shape` +
                ` of [batch, seq, embed_dim], got ${JSON.stringify(inputShape)}`)
        }

        const [decoderInputShape] = input_shapes;

        if (decoderInputShape?.length != 3) {
            throw Error(`${this.getClassName()}::build ${this.name} expects an input shape` +
                ` of [batch, seq, embed_dim], got ${JSON.stringify(inputShape)}`)
        }

        // initialize causal self attention sub-block's weights
        this.causalSelfAttention.build(decoderInputShape);
        this.causalSelfAttentionNorm.build(this.causalSelfAttention.computeOutputShape(decoderInputShape));

        // initialize feedforward sub-block's weights
        const feedforward1OutputShape = this.feedforward1.computeOutputShape(decoderInputShape);
        const feedforward2OutputShape = this.feedforward2.computeOutputShape(feedforward1OutputShape);

        this.feedforward1.build(decoderInputShape);
        this.feedforward2.build(feedforward1OutputShape);
        this.feedFowardNorm.build(feedforward2OutputShape);

        // track sublayers' weights
        this.trainableWeights = [
            ...this.causalSelfAttention.trainableWeights,
            ...this.causalSelfAttentionDropout.trainableWeights,
            ...this.causalSelfAttentionNorm.trainableWeights,
            ...this.feedforward1.trainableWeights,
            ...this.feedforward2.trainableWeights,
            ...this.feedForwardDropout.trainableWeights,
            ...this.feedFowardNorm.trainableWeights
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
     * Save the layer's hyperparameters for serialization
     */
    override getConfig() {
        const base_config = super.getConfig();

        const config = {
            numHeads: this.numHeads,
            embedDim: this.embedDim,
            useBias: this.useBias,
            dropout: this.dropout,
            activation: this.activation,
            dimsFeedForward: this.dimsFeedForward
        }

        Object.assign(config, base_config);

        return config;
    }

}


tf.serialization.registerClass(TransformerDecoder);
