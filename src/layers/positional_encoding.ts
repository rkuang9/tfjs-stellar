import * as tf from "@tensorflow/tfjs";
import { type LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { type Kwargs } from "@tensorflow/tfjs-layers/dist/types";


export interface PositionalEncodingArgs extends LayerArgs {
    // embedding size of each word/token, aka d_model from the paper
    embedDim: number;
    // the max length of each sentence, any more or less are truncated or padded
    maxSequenceLength?: number;
}


/**
 * This class implements the position encoding logic described in the
 * 2017 paper "Attention Is All You Need".
 * 
 * This layer is untrainable and accepts inputs of shape `[ batch, sequences, embedding dims ]`
 * and adds positional encoding to return an output tensor of the same shape.
 * 
 * @param embedDim the size of each token/word's embedding
 * @param maxSequenceLength the max number of tokens (words) per input (sentence), default `5120`
 */
export class PositionalEncoding extends tf.layers.Layer {
    static className = "PositionalEncoding";
    private readonly maxSequenceLength: number;
    private readonly embedDim: number;
    private positionalEncodings: tf.LayerVariable;


    constructor(args: PositionalEncodingArgs) {
        super(args);

        this.maxSequenceLength = args.maxSequenceLength ?? 5120;
        this.embedDim = args.embedDim;

        if (this.maxSequenceLength < 1) {
            throw Error(`${this.getClassName()}::constructor ${this.name} maxSequenceLength` +
                ` (${args.maxSequenceLength}) must be greater than 0`);
        }

        if (this.embedDim < 1) {
            throw Error(`${this.getClassName()}::constructor ${this.name} embedDim` +
                ` (${args.embedDim}) must be greater than 0`);
        }

        // positional encodings are not trainable
        this.positionalEncodings = this.addWeight('positional_encodings',
            [this.maxSequenceLength, this.embedDim], "float32",
            tf.initializers.zeros(), undefined, false);
    }


    /**
     * Forward propagation. Injects positional encoding to the input embeddings
     */
    override call(inputs: tf.Tensor | tf.Tensor[], kwargs: Kwargs): tf.Tensor | tf.Tensor[] {
        // validate the input tensors
        const input = Array.isArray(inputs) ? inputs[0] : inputs;
        const sequences = input.shape[1]!;

        if (input.shape.length != 3 || input.shape[2] != this.embedDim) {
            throw Error(`${this.getClassName()}::call ${this.name} expected an input shape of` +
                ` [batch, (up to ${this.maxSequenceLength}), ${this.embedDim}], instead got ${input.shape}`);
        }

        if (sequences > this.maxSequenceLength) {
            // unexpected sequence length
            throw Error(`${this.getClassName()}::call ${this.name} received an input with` +
                ` sequence length (${sequences}) which is greater than the max sequence length` +
                ` ${this.maxSequenceLength}`);
        }

        // perform forward propagation
        return tf.tidy(() => {
            return input.add(this.positionalEncodings.read()
                .slice([0, 0], [sequences, this.embedDim]) // gets the first "sequences" rows
                .expandDims(0)); // introduce the batch dimension and let add() broadcast it
        })
    }

    /**
     * Generate the positional encoding from the paper Attention Is All You Need.
     * Note that because the inner term of the position formula is the same for both even
     * and odd indices, we only create half of it and apply sine and cosine individually.
     */
    override build(inputShape: tf.Shape | tf.Shape[]): void {
        tf.tidy(() => {
            const embedDimHalved = Math.ceil(this.embedDim / 2);

            // create the position matrix as [ 0, 1, 2, 3, etc ],
            // and broadcast it horizontally to match the number of embeddings,
            const numerator = tf.range(0, this.maxSequenceLength, 1)
                .reshape([this.maxSequenceLength, 1])
                // this creates an extra, unsued positional encoding column later on for odd embedding sizes
                .broadcastTo([this.maxSequenceLength, embedDimHalved]);

            // the inner term's denominator's exponent's numerator is created as
            // [ 0, 0, 2, 2, 4, 4, etc ] ( technically [0, 2, 4] as explained above ) and not
            // [ 0, 2, 4, 6, 8, 10, etc ] because the even and odd indices are counted as pairs
            // when incrementing "i",
            // the denominator formula is 10_000^(2i/d_model) where each "i" is a sine cosine pair
            const denominator = tf.pow(10_000, tf.range(0, this.embedDim, 2).div(this.embedDim));

            const inner_term = numerator.div(denominator);

            const sine = tf.sin(inner_term);
            const cosine = tf.cos(inner_term);

            // horizontally interweave the sine and cosine columns together to form
            // [sin, cos, sin, cos, etc]
            // [sin, cos, sin, cos, etc]
            // etc
            const interweaved = [];
            const ALL_ROWS = -1;
            const ONE_COL = 1;
            const FIRST_ROW = 0;

            for (let targetCol = 0; targetCol < this.embedDim / 2; targetCol++) {
                interweaved.push(sine.slice([FIRST_ROW, targetCol], [ALL_ROWS, ONE_COL]))

                if (targetCol != Math.floor(this.embedDim / 2)) {
                    // for odd numbered embedDim sizes skip the last cosine column
                    // e.g. if embedDim = 5, create [ i=0 (sin), i=0 (cos), i=1 (sin), i=1 (cos), i=2 (sin) ]
                    // and the final i=2 (cos) is ignored
                    interweaved.push(cosine.slice([FIRST_ROW, targetCol], [ALL_ROWS, ONE_COL]))
                }
            }

            // add the positional encoding
            this.setWeights([tf.concat(interweaved, 1)]);
        });

        super.build(inputShape);
    }


    override computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        return inputShape;
    }


    override getConfig(): tf.serialization.ConfigDict {
        const base_config = super.getConfig();

        const config = {
            maxSequenceLength: this.maxSequenceLength,
            embedDim: this.embedDim,
        }

        Object.assign(config, base_config);

        return config;
    }
}


tf.serialization.registerClass(PositionalEncoding);
