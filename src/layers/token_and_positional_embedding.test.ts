import * as tf from '@tensorflow/tfjs';

import { TokenAndPositionalEmbedding } from '../layers/token_and_positional_embedding';

// disables warning for using the faster node backend,
// https://github.com/tensorflow/tfjs/issues/5349#issuecomment-885170504
tf.env().set('IS_NODE', false);


describe("PositionalEncoding tests", () => {
    test("layer initialization", () => {
        expect(() => new TokenAndPositionalEmbedding({ maxSequenceLength: 0, embedDim: 10, vocabularySize: 10_000 })).toThrow();
        expect(() => new TokenAndPositionalEmbedding({ embedDim: 0, vocabularySize: 10_000 })).toThrow();
        expect(() => new TokenAndPositionalEmbedding({ embedDim: 10, vocabularySize: 0 })).toThrow();

        expect(() => new TokenAndPositionalEmbedding({ embedDim: 10, vocabularySize: 10_000 })).not.toThrow();
        expect(() => new TokenAndPositionalEmbedding({ embedDim: 10, vocabularySize: 10_000 })).not.toThrow();
    })


    test("successfull forward calls", () => {
        const embed_dims = 32;
        const sequences = 4;
        const vocab_size = 10_000;
        const input = tf.randomUniform([2, sequences]);

        const embedding = new TokenAndPositionalEmbedding({ embedDim: embed_dims, dropout: 0.1, vocabularySize: vocab_size });
        expect(() => embedding.apply(input)).not.toThrow();
        expect(() => embedding.apply([input])).not.toThrow();
    })


    test("layer build", () => {
        const input_ok = tf.randomUniform([2, 4]);
        const input_too_many_words = tf.randomUniform([2, 700]);
        const input_is_image = tf.randomUniform([1, 32, 32, 3]);

        let embedding = new TokenAndPositionalEmbedding({ embedDim: 32, maxSequenceLength: 500, vocabularySize: 1_000 });
        expect(() => embedding.build(input_ok.shape)).not.toThrow();

        embedding = new TokenAndPositionalEmbedding({ embedDim: 32, maxSequenceLength: 500, vocabularySize: 1_000 });
        expect(() => embedding.build([input_ok.shape, input_ok.shape])).not.toThrow();

        new TokenAndPositionalEmbedding({ embedDim: 32, maxSequenceLength: 500, vocabularySize: 1_000 });
        expect(() => embedding.build(input_too_many_words.shape)).toThrow();
        expect(() => embedding.build(input_is_image.shape)).toThrow();
    })


    it("should throw when more than one input provided, input sequences are too large, or incorrect input rank", () => {
        const sequences_too_long = tf.randomUniform([10, 1000]);
        const multiple_correct_inputs = [tf.randomUniform([2, 3]), tf.randomUniform([2, 3])];
        const wrong_rank = tf.randomUniform([10, 32, 32]);

        const positional = new TokenAndPositionalEmbedding({ maxSequenceLength: 10, embedDim: 32, vocabularySize: 10_000 });
        positional.build([2, 3]); // get past the initial build call to test forward prop

        expect(() => positional.apply(sequences_too_long)).toThrow();
        expect(() => positional.apply(multiple_correct_inputs)).toThrow();
        expect(() => positional.apply(wrong_rank)).toThrow();
    })


    it("should return a non-empty config dict", () => {
        const embedding = new TokenAndPositionalEmbedding({ embedDim: 32, vocabularySize: 10_000 });
        expect(Object.keys(embedding.getConfig())).not.toBe(0);
    })


    it("should return an output shape of [batch, sequences, embed dims]", () => {
        const words = 100;
        const batch = 2;
        const embed_dims = 64;

        const input = tf.randomUniform([batch, words]);

        const embedding = new TokenAndPositionalEmbedding({ embedDim: embed_dims, vocabularySize: 10_000 });

        expect(embedding.computeOutputShape(input.shape)).toEqual([batch, words, embed_dims]);
    })
});
