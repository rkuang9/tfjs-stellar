import * as tf from "@tensorflow/tfjs";

import { TransformerEncoder } from "@/layers/transformer_encoder";

// disables warning for using the faster node backend,
// https://github.com/tensorflow/tfjs/issues/5349#issuecomment-885170504
tf.env().set('IS_NODE', false);


describe("TransformerEncoder tests", () => {
    it("should return an output with the same shape as the input", () => {
        const input = tf.randomUniform([2, 3, 10]);

        const decoder = new TransformerEncoder({
            numHeads: 2, embedDim: input.shape.at(-1)!,
            dropout: 0.5, activation: "gelu", dimsFeedForward: 512, useBias: true
        });

        const output = decoder.apply(input) as tf.Tensor;

        expect(output.shape.length).toBe(input.shape.length);
    })


    test("correct forward calls", () => {
        const input = tf.randomUniform([2, 3, 10]);

        const encoder = new TransformerEncoder({ numHeads: 2, embedDim: input.shape.at(-1)! });
        expect(() => encoder.apply(input)).not.toThrow();
        expect(() => encoder.apply([input])).not.toThrow();

        const causal = new TransformerEncoder({ numHeads: 2, embedDim: input.shape.at(-1)!, causal: true });
        expect(() => causal.apply(input)).not.toThrow();
        expect(() => causal.apply([input])).not.toThrow();
    })


    it("should fail to instantiate a layer if heads count is not divisible by the input's embedding dimension", () => {
        const input = tf.randomUniform([2, 3, 10]);

        expect(() => new TransformerEncoder({ numHeads: 3, embedDim: input.shape.at(-1)! })).toThrow();
        expect(() => new TransformerEncoder({ numHeads: 5, embedDim: input.shape.at(-1)! })).not.toThrow();
    })


    it("should not accept non-rank 3 tensor inputs", () => {
        const incorrect_input = tf.randomUniform([2, 3, 10, 10]);
        const incorrect_input2 = tf.randomUniform([2, 3]);
        const correct_input = tf.randomUniform([2, 3, 10]);


        const encoder = new TransformerEncoder({ numHeads: 2, embedDim: incorrect_input.shape.at(-1)! });
        expect(() => encoder.apply([correct_input, correct_input])).toThrow();

        expect(() => encoder.apply(incorrect_input)).toThrow();
        expect(() => encoder.apply(incorrect_input2)).toThrow();

        expect(() => encoder.apply([correct_input, incorrect_input])).toThrow();
        expect(() => encoder.apply([incorrect_input, correct_input])).toThrow();

        expect(() => encoder.apply([correct_input, incorrect_input2])).toThrow();
        expect(() => encoder.apply([incorrect_input2, correct_input])).toThrow();
    })


    it("should accept exactly one input", () => {
        const input = tf.randomUniform([2, 3, 10]);

        const encoder = new TransformerEncoder({ numHeads: 1, embedDim: input.shape.at(-1)! });
        expect(() => encoder.apply(input)).not.toThrow();
        expect(() => encoder.apply([input])).not.toThrow();

        expect(() => encoder.apply([])).toThrow();
        expect(() => encoder.apply([input, input])).toThrow();
        expect(() => encoder.apply([input, input, input])).toThrow()
    })


    it("should return a non-empty config dict", () => {
        const input = tf.randomUniform([2, 3, 10]);

        const encoder = new TransformerEncoder({ numHeads: 1, embedDim: input.shape.at(-1)! });
        expect(Object.keys(encoder.getConfig())).not.toBe(0);
    })
})
