import * as tf from '@tensorflow/tfjs';
import { type LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { type Kwargs } from '@tensorflow/tfjs-layers/dist/types';

import { PositionalEncoding, type PositionalEncodingArgs } from '../layers/positional_encoding';


export interface TokenAndPositionalEmbeddingArgs extends LayerArgs, PositionalEncodingArgs {
    vocabularySize: number;
    dropout?: number
}


/**
 * This class implements combines sinusoidal positional encoding from the
 * 2017 paper "Attention Is All You Need" with a normal embedding layer to
 * form a simplified single embedding layer.
 * 
 * This layer accepts tokenized inputs of the shape `[ batch, tokens ]` and runs
 * it through an embedding layer before adding sinusoidal positional encoding.
 * 
 * @param embedDim the size of each token/word's embedding
 * @param vocabularySize the number of tokens to embed
 * @param maxSequenceLength the max number of tokens (words) per input (sentence), default `5120`
 * @param dropout applies dropout to the positionally encoded embeddings, default `0.1`
 */
export class TokenAndPositionalEmbedding extends tf.layers.Layer {
    static className = "TokenAndPositionalEmbedding";

    private readonly embedDim: number;
    private readonly vocabularySize: number;
    private embedding: tf.layers.Layer;

    private positional: tf.layers.Layer
    private readonly maxSequenceLength: number;
    private readonly dropout: number;

    private dropoutLayer: tf.layers.Layer;


    constructor({ embedDim, vocabularySize, maxSequenceLength, dropout, ...args }: TokenAndPositionalEmbeddingArgs) {
        super(args);

        this.embedDim = embedDim;
        this.vocabularySize = vocabularySize;
        this.maxSequenceLength = maxSequenceLength ?? 5120;
        this.dropout = dropout ?? 0.1;

        if (this.dropout >= 1) {
            throw Error(`${this.getClassName()}::constructor dropout must be within [0, 1)`);
        }

        this.embedding = tf.layers.embedding({
            inputDim: this.vocabularySize,
            outputDim: this.embedDim,
        });

        this.positional = new PositionalEncoding({
            maxSequenceLength: this.maxSequenceLength,
            embedDim: this.embedDim,
        });

        this.dropoutLayer = tf.layers.dropout({ rate: this.dropout });
    }


    /**
     * Forward propagation. 
     */
    override call(inputs: tf.Tensor | tf.Tensor[], kwargs: Kwargs) {
        if (Array.isArray(inputs) && inputs.length != 1) {
            throw Error(`${this.getClassName()}::call ${this.name} expects exactly` +
                ` 1 tensor input, received ${inputs.length}`);
        }

        return tf.tidy(() => {
            let output = this.positional.apply(this.embedding.apply(inputs)) as tf.Tensor;
            output = this.dropoutLayer.apply(output) as tf.Tensor;

            return output;
        })
    }


    /**
     * Build the sublayers and enable serialization
     */
    override build(inputShape: tf.Shape | tf.Shape[]): void {
        let input_shapes: tf.Shape[] = [];

        // only consider the first shape if multiple provided
        if (Array.isArray(inputShape) && Array.isArray(inputShape[0])) {
            // input is an array of shapes
            input_shapes = inputShape as tf.Shape[];
        } else if (inputShape.length != 0) {
            // input is a single shape
            input_shapes = [inputShape as tf.Shape];
        }

        if (input_shapes[0].length != 2 || input_shapes[0][1]! > this.maxSequenceLength) {
            throw Error(`${this.getClassName()}::build ${this.name} expected an input of` +
                ` shape [batch, tokens] where tokens < ${this.maxSequenceLength},` +
                ` received ${JSON.stringify(input_shapes[0])}`);
        }

        // initialize the sublayers' weights
        this.embedding.build(input_shapes[0]);
        this.positional.build(this.embedding.computeOutputShape(input_shapes[0]));

        // no need to rename weights, haven't found a case where their names collide
        this.trainableWeights = [
            ...this.embedding.trainableWeights,
            ...this.positional.trainableWeights
        ];

        super.build(input_shapes[0]);
    }


    /**
     * The output shape, for an input shape of [batch, sequences], is
     * [batch, sequences, embedDim]
     */
    override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        const embedding_shape = this.embedding.computeOutputShape(inputShape);
        const positional_shape = this.positional.computeOutputShape(embedding_shape);

        return positional_shape;
    }


    override getConfig(): tf.serialization.ConfigDict {
        const base_config = super.getConfig();

        const config = {
            embedDim: this.embedDim,
            vocabularySize: this.vocabularySize,
            maxSequenceLength: this.maxSequenceLength,
            dropout: this.dropout,
        }

        Object.assign(config, base_config);

        return config;
    }
}


tf.serialization.registerClass(TokenAndPositionalEmbedding);
