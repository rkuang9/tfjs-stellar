import * as tf from "@tensorflow/tfjs";
import { type Kwargs } from "@tensorflow/tfjs-layers/dist/types";
import { type ActivationIdentifier } from "@tensorflow/tfjs-layers/dist/keras_format/activation_config";

import { MultiHeadAttention, type MultiHeadAttentionArgs } from "../layers/multihead_attention";


export interface TransformerEncoderArgs extends MultiHeadAttentionArgs {
    activation?: "relu" | "gelu";
    dimsFeedForward?: number;
}


/**
 * This class implements the transformer encoder architecture from the 2017 paper
 * Attention Is All You Need.
 * 
 * This layer accepts exactly one tensor input with the shape
 * `[ batch, sequences, embedding dims ]`.
 * 
 * @param numHeads number of attention heads to use
 * @param embedDim the embedding size of the input (input embeddings, typically the last dimension)
 * @param causal use causal masking, default `false` for encoders
 * @param dropout use dropout during the attention calculations, default `0.1`
 * @param activation the activation of the intermediate feed forward layer, default `relu`
 * @param dimsFeedForward the size of the intermediate feed forward layer, default `2048`
 * @param useBias use bias for the dense sublayers and multiHead attention's dense sublayers, default `true`
 */
export class TransformerEncoder extends tf.layers.Layer {
    static className = "TransformerEncoder";

    private readonly selfAttention: tf.layers.Layer;
    private readonly selfAttentionDropout: tf.layers.Layer;
    private readonly selfAttentionNorm: tf.layers.Layer;

    private readonly reluLayer: tf.layers.Layer;
    private readonly linearLayer: tf.layers.Layer;
    private readonly feedForwardDropout: tf.layers.Layer;
    private readonly feedFowardNorm: tf.layers.Layer;

    private readonly numHeads: number;
    private readonly embedDim: number;
    private readonly causal: boolean;
    private readonly useBias: boolean;
    private readonly dropout: number;
    private readonly activation: ActivationIdentifier;
    private readonly dimsFeedForward: number;


    constructor({ numHeads, embedDim, causal, useBias, dropout, activation, dimsFeedForward, ...args }: TransformerEncoderArgs) {
        super(args);

        this.numHeads = numHeads;
        this.embedDim = embedDim;
        this.causal = causal ?? false;
        this.useBias = useBias ?? true;
        this.dropout = dropout ?? 0.1;
        this.activation = activation ?? "relu";
        this.dimsFeedForward = dimsFeedForward ?? 2048;

        if (this.dropout >= 1) {
            throw Error(`${this.getClassName()}::constructor dropout must be within [0, 1)`);
        }

        // self attention sub-block
        this.selfAttention = new MultiHeadAttention({
            numHeads: this.numHeads, embedDim: this.embedDim, useBias: this.useBias,
            dropout: this.dropout, causal: this.causal
        });
        this.selfAttentionDropout = tf.layers.dropout({ rate: this.dropout })
        this.selfAttentionNorm = tf.layers.layerNormalization({ epsilon: 1e-6 });

        // feed forward sub-block
        this.reluLayer = tf.layers.dense({
            units: this.dimsFeedForward, activation: this.activation,
            useBias: this.useBias
        });
        this.linearLayer = tf.layers.dense({
            units: this.embedDim, activation: "linear",
            useBias: this.useBias
        });
        this.feedForwardDropout = tf.layers.dropout({ rate: this.dropout });
        this.feedFowardNorm = tf.layers.layerNormalization({ epsilon: 1e-6 });
    }


    /**
     * Forward propagation
     */
    override call(inputs: tf.Tensor | tf.Tensor[], kwargs: Kwargs): tf.Tensor | tf.Tensor[] {
        // validate the input tensors
        let input: tf.Tensor;

        if (Array.isArray(inputs)) {
            if (inputs.length != 1) {
                throw Error(`${this.getClassName}::call ${this.name} expects exactly 1 tensor` +
                    ` input, got ${inputs.length} inputs instead.`);
            }

            input = inputs[0];
        } else {
            input = inputs;
        }

        // perform forward propagation
        return tf.tidy(() => {
            const attention = this.selfAttentionBlock(input, kwargs);
            const feedforward = this.feedForwardBlock(attention, kwargs);

            return feedforward;
        });
    }


    private selfAttentionBlock(x: tf.Tensor, kwargs: Kwargs): tf.Tensor {
        return tf.tidy(() => {
            const residual = x;

            let attention = this.selfAttention.apply(x, kwargs) as tf.Tensor;
            attention = this.selfAttentionDropout.apply(attention, kwargs) as tf.Tensor;
            attention = tf.add(attention, residual);
            attention = this.selfAttentionNorm.apply(attention) as tf.Tensor;

            return attention;
        });
    }


    private feedForwardBlock(x: tf.Tensor, kwargs: Kwargs): tf.Tensor {
        return tf.tidy(() => {
            const residual = x;

            let feedForward = this.reluLayer.apply(x);
            feedForward = this.linearLayer.apply(feedForward);
            feedForward = this.feedForwardDropout.apply(feedForward, kwargs) as tf.Tensor;
            feedForward = tf.add(feedForward, residual);
            feedForward = this.feedFowardNorm.apply(feedForward) as tf.Tensor;

            return feedForward;
        });
    }


    /**
     * Initialize the sublayers' weights and track them to enable backpropagation.
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

        // expects only 1 rank 3 tensor input
        if (input_shapes.length != 1 || input_shapes[0].length != 3) {
            throw Error(`${this.getClassName()}::build ${this.name} expects a single input shape of [batch, seq, embed_dim], got ${JSON.stringify(inputShape)}`)
        }

        // initialize self attention sub-block's weights
        this.selfAttention.build(inputShape);
        this.selfAttentionNorm.build(inputShape);

        // inintialize feedforward sub-block's weights
        const reluLayerOutputShape = this.reluLayer.computeOutputShape(inputShape);
        const linearLayerOutputShape = this.linearLayer.computeOutputShape(reluLayerOutputShape);

        this.reluLayer.build(inputShape);
        this.linearLayer.build(reluLayerOutputShape);
        this.feedFowardNorm.build(linearLayerOutputShape);

        // track sublayers' weights
        this.trainableWeights = [
            ...this.selfAttention.trainableWeights,
            ...this.selfAttentionDropout.trainableWeights,
            ...this.selfAttentionNorm.trainableWeights,
            ...this.reluLayer.trainableWeights,
            ...this.linearLayer.trainableWeights,
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
    override getConfig(): tf.serialization.ConfigDict {
        const base_config = super.getConfig();

        const config = {
            numHeads: this.numHeads,
            embedDim: this.embedDim,
            causal: this.causal,
            useBias: this.useBias,
            dropout: this.dropout,
            activation: this.activation,
            dimsFeedForward: this.dimsFeedForward
        };

        Object.assign(config, base_config);

        return config;
    }
}


tf.serialization.registerClass(TransformerEncoder);
