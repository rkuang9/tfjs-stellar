import * as tf from '@tensorflow/tfjs';

import { TransformerDecoder } from '../layers/transformer_decoder';

// disables warning for using the faster node backend,
// https://github.com/tensorflow/tfjs/issues/5349#issuecomment-885170504
tf.env().set('IS_NODE', false);


describe("TransformerDecoder tests", () => {
    it("should return an output with the same shape as the input", () => {
        const input = tf.randomUniform([2, 3, 12]);

        const decoder = new TransformerDecoder({
            numHeads: 2, embedDim: input.shape.at(-1)!,
            dropout: 0.5, activation: "gelu", dimsFeedForward: 321, useBias: false
        });

        const output = decoder.apply(input) as tf.Tensor;

        expect(output.shape.length).toBe(input.shape.length);
    })


    test("forward calls", () => {
        const input = tf.randomUniform([2, 3, 12]);
        const mask = tf.randomUniform([input.shape[0]!, input.shape[1]!], -1, 2, "bool");
        const incorrect_mask = tf.randomUniform([2, 5, 12], -1, 2, "bool");


        const decoder = new TransformerDecoder({ numHeads: 2, embedDim: input.shape.at(-1)! });
        expect(() => decoder.apply(input)).not.toThrow();
        expect(() => decoder.apply([input])).not.toThrow();

        // causal masking
        const causal = new TransformerDecoder({ numHeads: 2, embedDim: input.shape.at(-1)!, causal: true });
        expect(() => causal.apply(input)).not.toThrow();
        expect(() => causal.apply([input])).not.toThrow();
    })


    it("should fail to instantiate a layer if heads count is not divisible by the input's embedding dimension", () => {
        const input = tf.randomUniform([2, 3, 12]);

        expect(() => new TransformerDecoder({ numHeads: 3, embedDim: input.shape.at(-1)! })).not.toThrow();
        expect(() => new TransformerDecoder({ numHeads: 5, embedDim: input.shape.at(-1)! })).toThrow();
    })


    it("should not accept non-rank 3 tensor inputs", () => {
        const embed_dim = 12;

        const BAD_RANK4 = tf.randomUniform([2, 3, 12, embed_dim]);
        const BAD_RANK2 = tf.randomUniform([2, embed_dim]);
        const GOOD = tf.randomUniform([2, 3, embed_dim]);
        const mask = tf.randomUniform([GOOD.shape[0]!, GOOD.shape[1]!], -1, 2, "bool");

        let decoder = new TransformerDecoder({ numHeads: 2, embedDim: embed_dim });

        // BAD
        expect(() => decoder.apply(BAD_RANK4)).toThrow();
        expect(() => decoder.apply(BAD_RANK2)).toThrow();

        // OK
        decoder = new TransformerDecoder({ numHeads: 2, embedDim: embed_dim });
        expect(() => decoder.apply(GOOD)).not.toThrow();
        expect(() => decoder.apply([GOOD])).not.toThrow();
        expect(() => decoder.apply([GOOD, mask])).not.toThrow();
    })


    it("should not accept inputs that are less or more than 1 and 2 tensors", () => {
        const input = tf.randomUniform([2, 3, 12]);

        let decoder = new TransformerDecoder({ numHeads: 1, embedDim: input.shape.at(-1)! });
        // OK
        expect(() => decoder.apply(input)).not.toThrow();
        expect(() => decoder.apply([input])).not.toThrow();

        // BAD
        decoder = new TransformerDecoder({ numHeads: 1, embedDim: input.shape.at(-1)! });
        expect(() => decoder.apply([])).toThrow(); // stops at build()
        decoder.apply(input); // get past the initial build
        expect(() => decoder.apply([input, input, input])).toThrow();
        expect(() => decoder.apply([input, input, input, input])).toThrow();

        // BAD (tests build())
        decoder = new TransformerDecoder({ numHeads: 1, embedDim: input.shape.at(-1)! });
        expect(() => decoder.apply([input, input, input])).toThrow();
        expect(() => decoder.apply([input, input, input, input])).toThrow();
    })


    it("should return a non-empty config dict", () => {
        const input = tf.randomUniform([2, 3, 12]);

        const decoder = new TransformerDecoder({ numHeads: 1, embedDim: input.shape.at(-1)! });
        expect(Object.keys(decoder.getConfig())).not.toBe(0);
    })
})
