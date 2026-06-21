import * as tf from "@tensorflow/tfjs";
import { type LayerArgs } from "@tensorflow/tfjs-layers/dist/engine/topology";


export function applyRope(x: tf.Tensor, dim: number, cosine_cache: tf.Tensor, sine_cache: tf.Tensor) {
    return tf.tidy(() => {
        const seq_length = x.shape[2]!;

        // get a slice of the pre-computed cache, up to the input's sequence length
        const cosine = cosine_cache.slice([0, 0, 0, 0], [1, 1, seq_length, dim]);
        const sine = sine_cache.slice([0, 0, 0, 0], [1, 1, seq_length, dim]);

        // apply RoPE formula (x1 * cosine) + (rotate(-x2) * sine)
        const rotated_x = rotateHalf(x, dim);

        return tf.add(tf.mul(x, cosine), tf.mul(rotated_x, sine));
    });
}


export function rotateHalf(x: tf.Tensor, dim: number): tf.Tensor {
    return tf.tidy(() => {
        // reshape the last dimension such that adjacent coordinates are paired together
        // [x1, x2, x3, x4] -> [[x1, x2], [x3, x4]]
        // the leading dimensions are flattened because TFJS has issues during
        // backpropagation with 5D slicing
        const reshaped = x.reshape([-1, dim / 2, 2]);

        const x1 = reshaped.slice([0, 0, 0], [-1, -1, 1]);
        const x2 = reshaped.slice([0, 0, 1], [-1, -1, 1]);

        // [x1, x2] -> [-x2, x1]
        const rotated = tf.concat([tf.neg(x2), x1], -1);

        return rotated.reshape(x.shape);
    });
}


export function createRoPECache(dim: number, max_sequence_length: number, theta: number = 10_000) {
    return tf.tidy(() => {
        // [dim]
        const inv_frequencies = tf.div<tf.Tensor1D>(1, tf.pow(
            theta,
            tf.range(0, Math.floor(dim / 2) * 2, 2, "float32").div(dim)));

        // [max_sequene_length]
        const sequence_indices = tf.range(0, max_sequence_length);
        // 
        const freq = tf.outerProduct(sequence_indices, inv_frequencies);

        // cache final shape [max_sequence_length, dim]
        const freq_pairs = tf.stack([freq, freq], -1)
            .reshape([max_sequence_length, dim]);

        return [
            tf.keep(tf.cos(freq_pairs).expandDims(0).expandDims(0)),
            tf.keep(tf.sin(freq_pairs).expandDims(0).expandDims(0))
        ]
    });
}


export interface RotaryPositionEmbeddingArgs extends LayerArgs {
    /**
     * The dimension of each head (rounded down), e.g. `Math.floor(embedDim / numHeads)`
     */
    dim: number,
    /**
     * The RoPE cache will be pre-calculated up to the max sequence length, and re-caculated as needed. Defaults to `4096`.
     */
    maxSequenceLength?: number,
    /**
     * The base for the geometric progression used to compute the rotation angles. Defaults to `10_000`.
     */
    theta?: number,
}


/**
 * Implements RoPE from the RoFormer: Enhanced Transformer with Rotary Position Embedding paper
 * Inspired by: https://meta-pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
 */
export class RotaryPositionEmbedding extends tf.layers.Layer {
    static className = "RotaryPositionEmbedding";

    protected dim: number;
    protected max_sequence_length: number;
    protected theta: number;

    // cached sine and cosine frequencies, untrainable weights
    protected cosine_cache: tf.LayerVariable;
    protected sine_cache: tf.LayerVariable;

    constructor({ dim, maxSequenceLength = 4096, theta = 10_000, ...args }: RotaryPositionEmbeddingArgs) {
        super(args);

        if (dim % 2 !== 0) {
            throw Error(`${this.getClassName()}::constructor ${this.name} expected dim to be even, got ${dim}`);
        }

        this.dim = dim;
        this.max_sequence_length = maxSequenceLength;
        this.theta = theta;

        this.cosine_cache = this.addWeight("sine_cache",
            [1, 1, maxSequenceLength, Math.floor(this.dim)],
            "float32", tf.initializers.zeros(), undefined, false);

        this.sine_cache = this.addWeight("cosine_cache",
            [1, 1, maxSequenceLength, Math.floor(this.dim)],
            "float32", tf.initializers.zeros(), undefined, false);
    }


    override call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor | tf.Tensor[] {
        const shape = Array.isArray(inputs) ? inputs[0].shape : inputs.shape;
        const seq_length = shape[2];

        if (seq_length > this.max_sequence_length) {
            // expand cache to the nearest power of 2
            this.max_sequence_length = Math.pow(2, Math.ceil(Math.log2(seq_length)));
            this.build([]);
        }

        return applyRope(
            Array.isArray(inputs) ? inputs[0] : inputs,
            this.dim,
            this.cosine_cache.read(),
            this.sine_cache.read())
    }


    override build(input_shape: tf.Shape | tf.Shape[]) {
        const [cosine, sine] = createRoPECache(
            this.dim, this.max_sequence_length, this.theta);

        this.cosine_cache.dispose();
        this.sine_cache.dispose();

        this.cosine_cache = new tf.LayerVariable(cosine);
        this.sine_cache = new tf.LayerVariable(sine);

        this.nonTrainableWeights = [
            new tf.LayerVariable(cosine),
            new tf.LayerVariable(sine)
        ];

        this.setWeights([cosine, sine]);
    }


    /**
     * Output shape: [batch, head, sequence, head_dim]
     */
    public computeOutputShape(input_shape: tf.Shape | tf.Shape[]) {
        return Array.isArray(input_shape[0])
            ? input_shape[0] as tf.Shape
            : input_shape as tf.Shape;
    }
}

tf.serialization.registerClass(RotaryPositionEmbedding);
