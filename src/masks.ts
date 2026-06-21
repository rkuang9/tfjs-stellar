import * as tf from "@tensorflow/tfjs";


/**
 * Generate a causal mask used in self-attention to prevent tokens from looking
 * ahead. The values in the upper right portion of the mask matrix are set to
 * -1e7 so that they have no impact during scaled dot product attention.
 */
export function causal(query_seq_length: number, key_seq_length: number) {
    return tf.linalg.bandPart(tf.ones([query_seq_length, key_seq_length]), -1, 0)
        .sub(1)
        .mul(1e7);
}


/**
 * Generate a self-attention mask that prevents packed sequences from cross document
 * boundaries and attending to each other. The result is a tensor of diagonally
 * positioned blocks of zeroes (allow attention) and -1e7 values (prevent attention).
 * The latter is scored zero during the scaled dot product attention's softmax operation.
 * 
 * @param boundaries an array of ones (denotes start of a new sample or docment) and zeroes
 * 
 * Example boundary of 3 samples that are packed into one:
 * `[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]`
 */
export function packing(boundaries: Int32Array) {
    // see images at
    // https://reddit.com/r/LocalLLaMA/comments/197efaz/training_llama_mistral_and_mixtralmoe_faster_with/
    return tf.tidy(() => {
        // cumsum transforms the tensor such that each sequence in the pack gets its own id,
        const partitions = tf.tensor1d(boundaries).cumsum();

        return partitions.expandDims(1)
            .equal(partitions.expandDims(0))
            .sub(1)
            .mul(1e7)
            // introduce a head dimension so it can be broadcasted
            .expandDims(0);
    })
}
