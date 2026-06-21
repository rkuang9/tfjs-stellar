import * as tf from '@tensorflow/tfjs';

import { CachedRoPEMultiHeadAttention } from '../layers/cached_rope_multihead_attention';
import { causal as generateCausalMask } from "../masks";
import { MultiHeadAttention } from '../layers/multihead_attention';

// disables warning for using the faster node backend,
// https://github.com/tensorflow/tfjs/issues/5349#issuecomment-885170504
tf.env().set('IS_NODE', false);


describe("MultiHeadAttention tests", () => {
    it("should fail to instantiate a layer if heads count is not divisible by the input's embedding dimension", () => {
        expect(() => new CachedRoPEMultiHeadAttention({ numHeads: 3, embedDim: 10 })).toThrow();
        expect(() => new CachedRoPEMultiHeadAttention({ numHeads: 15, embedDim: 60 })).not.toThrow();
    })


    test("successfull forward calls", () => {
        const input = tf.randomUniform([2, 3, 12]);

        const attention = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: input.shape.at(-1)! });
        expect(() => attention.apply(input)).not.toThrow();
        expect(() => attention.apply([input])).not.toThrow();

        const causal = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: input.shape.at(-1)!, causal: true });
        expect(() => causal.apply(input)).not.toThrow();
        expect(() => causal.apply([input])).not.toThrow();
    })


    test("query and value must have the same shape for scaled dot product attention to succeed", () => {
        const query = tf.randomUniform([2, 3, 12]);
        const key = tf.randomUniform([2, 3, 12]);
        const value = tf.randomUniform([2, 3, 12]);
        const value_thats_too_long = tf.randomUniform([2, 100, 12]);

        const attention = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: query.shape.at(-1)! });
        expect(() => attention.apply([query, key, value])).not.toThrow();
        expect(() => attention.apply([query, key, value_thats_too_long])).toThrow();
    })


    it("should only accept rank 3 tensors", () => {
        const embed_dims = 12;

        const BAD_RANK2 = tf.randomUniform([2, embed_dims]);
        const GOOD = tf.randomUniform([2, 3, embed_dims]);
        const BAD_RANK4 = tf.randomUniform([2, 3, 10, embed_dims]);

        const attention = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: embed_dims });

        // BAD
        expect(() => attention.apply(BAD_RANK2)).toThrow();
        expect(() => attention.apply([BAD_RANK2])).toThrow();
        expect(() => attention.apply([BAD_RANK2, BAD_RANK2, BAD_RANK2])).toThrow();

        // OK
        expect(() => attention.apply(GOOD)).not.toThrow();
        expect(() => attention.apply([GOOD])).not.toThrow();
        expect(() => attention.apply([GOOD, GOOD, GOOD])).not.toThrow();

        // BAD
        expect(() => attention.apply(BAD_RANK4)).toThrow();
        expect(() => attention.apply([BAD_RANK4])).toThrow();
        expect(() => attention.apply([BAD_RANK4, BAD_RANK4, BAD_RANK4])).toThrow();

        // BAD
        expect(() => attention.apply([GOOD, BAD_RANK2, BAD_RANK4])).toThrow();
        expect(() => attention.apply([BAD_RANK2, GOOD, BAD_RANK4])).toThrow();
        expect(() => attention.apply([BAD_RANK2, BAD_RANK4, GOOD])).toThrow();
        expect(() => attention.apply([BAD_RANK2, GOOD, GOOD])).toThrow();
        expect(() => attention.apply([GOOD, GOOD, BAD_RANK4])).toThrow();
    })


    it("should only 1 or 3 inputs total", () => {
        const input = tf.randomUniform([2, 3, 12]);

        let attention = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: input.shape.at(-1)! });

        // OK
        expect(() => attention.apply(input, { packingMask: undefined })).not.toThrow();
        expect(() => attention.apply([input])).not.toThrow();
        // reinitialize to rerun build()
        attention = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: input.shape.at(-1)! });
        expect(() => attention.apply([input, input, input])).not.toThrow();

        // BAD
        expect(() => attention.apply([])).toThrow();
        expect(() => attention.apply([input, input])).toThrow();
        // reinitialize to rerun build()
        attention = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: input.shape.at(-1)! });
        expect(() => attention.apply([input, input, input, input])).toThrow();
    })


    test("attention masking", () => {
        const query = tf.randomUniform([2, 3, 12]);
        const key = tf.randomUniform([2, 3, 12]);
        const value = tf.randomUniform([2, 3, 12]);

        const attention = new CachedRoPEMultiHeadAttention({ numHeads: 2, embedDim: query.shape.at(-1)!, causal: true });

        expect(() => attention.call(query, {})).not.toThrow();

        // cross attention
        expect(() => attention.call([query, key, value], {})).not.toThrow();


        const query5 = tf.randomUniform([2, 5, 10]);
        const key4 = tf.randomUniform([2, 4, 10]);
        const value5 = tf.randomUniform([2, 4, 10]);

        const expected_mask = tf.tensor([[
            // vertical represents query, false means that token cannot attend to the keys
            // horizontal represents key, false means that token cannot attend to the queries
            [false, false, false, false],
            [true, true, true, false,],
            [true, true, true, false,],
            [false, false, false, false],
            [true, true, true, false,],
        ]]);

        const packing_mask = tf.tensor([
            [0, 0, 0, -1e7, -1e7],
            [0, 0, 0, -1e7, -1e7],
            [0, 0, 0, -1e7, -1e7],
            [-1e7, -1e7, -1e7, 0, 0],
            [-1e7, -1e7, -1e7, 0, 0]
        ])

        // for causal attention, the attention mask must be boolean
        expect(() => MultiHeadAttention.scaledDotProductionAttention(query5, key4, value5, expected_mask.asType("float32"), null, null, 0.1, true, { scaling_factor: 10 })).toThrow();
        // for causal attention, using pre-calculated causal mask
        expect(() => MultiHeadAttention.scaledDotProductionAttention(query5, key4, value5, expected_mask.asType("float32"), null, generateCausalMask(query5.shape[1]!, key4.shape[1]!), 0.2, true, { scaling_factor: 10 })).toThrow();
        // when not using causal attention, the attention mask can be a float32 tensor
        expect(() => MultiHeadAttention.scaledDotProductionAttention(query5, key4, value5, expected_mask.asType("float32"), null, null, 0, false)).not.toThrow();
        // packing mask for self attention
        expect(() => MultiHeadAttention.scaledDotProductionAttention(query5, query5, query5, null, packing_mask, null, 0.9, true)).not.toThrow();
    })


    it("should return a non-empty config dict", () => {
        const input = tf.randomUniform([2, 3, 10]);

        const attention = new CachedRoPEMultiHeadAttention({ numHeads: 1, embedDim: input.shape.at(-1)! });
        expect(Object.keys(attention.getConfig())).not.toBe(0);
    })


    test("causal attention hard coded values", () => {
        // input and output shapes: [2, 3, 10]
        const input = tf.tensor([
            [[0.2109915, 0.6158954, 0.6012088, 0.9867562, 0.8728716, 0.7496274, 0.8173883, 0.2958342, 0.9650571, 0.2075207],
            [0.2946285, 0.9779906, 0.3203818, 0.4037617, 0.3762881, 0.9863171, 0.6655593, 0.7707329, 0.3216831, 0.7984023],
            [0.9080769, 0.0026282, 0.379492, 0.0162054, 0.1939302, 0.2201049, 0.8190675, 0.0203963, 0.0114392, 0.5015539]],

            [[0.6241482, 0.7631097, 0.6687831, 0.7259795, 0.0457698, 0.6889264, 0.0853676, 0.8697655, 0.3637198, 0.2105307],
            [0.5221761, 0.4476321, 0.1244729, 0.8863543, 0.7319002, 0.2954829, 0.3200496, 0.0905503, 0.607977, 0.1309131],
            [0.4693873, 0.4609751, 0.9170766, 0.7065565, 0.4795104, 0.3225758, 0.1353116, 0.7083887, 0.1928891, 0.967386]]
        ]);

        const expected = tf.tensor([
            [[0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344],
            [0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376],
            [0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539]],

            [[0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718],
            [0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268],
            [0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877]]
        ]);


        const attention = new CachedRoPEMultiHeadAttention({ numHeads: 1, embedDim: input.shape.at(-1)!, causal: true });
        attention.build(input.shape);
        attention.setWeights(attention.getWeights().map(weight => tf.onesLike(weight).mul(0.05)));

        expect(expected.sub(attention.apply(input) as tf.Tensor).sum().dataSync()[0]).toBeLessThan(1e-6);
    })


    test("non-causal attention hard coded values", () => {
        // input and output shapes: [2, 3, 10]
        const input = tf.tensor([
            [[0.2109915, 0.6158954, 0.6012088, 0.9867562, 0.8728716, 0.7496274, 0.8173883, 0.2958342, 0.9650571, 0.2075207],
            [0.2946285, 0.9779906, 0.3203818, 0.4037617, 0.3762881, 0.9863171, 0.6655593, 0.7707329, 0.3216831, 0.7984023],
            [0.9080769, 0.0026282, 0.379492, 0.0162054, 0.1939302, 0.2201049, 0.8190675, 0.0203963, 0.0114392, 0.5015539]],

            [[0.6241482, 0.7631097, 0.6687831, 0.7259795, 0.0457698, 0.6889264, 0.0853676, 0.8697655, 0.3637198, 0.2105307],
            [0.5221761, 0.4476321, 0.1244729, 0.8863543, 0.7319002, 0.2954829, 0.3200496, 0.0905503, 0.607977, 0.1309131],
            [0.4693873, 0.4609751, 0.9170766, 0.7065565, 0.4795104, 0.3225758, 0.1353116, 0.7083887, 0.1928891, 0.967386]]
        ]);


        const expected = tf.tensor([
            [[0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344, 0.2055344],
            [0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376, 0.205376],
            [0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539, 0.2042539]],

            [[0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718, 0.1966718],
            [0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268, 0.1966268],
            [0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877, 0.1966877]]
        ]);

        const attention = new CachedRoPEMultiHeadAttention({ numHeads: 1, embedDim: input.shape.at(-1)!, causal: false });
        attention.build(input.shape);
        attention.setWeights(attention.getWeights().map(weight => tf.onesLike(weight).mul(0.05)));

        expect(expected.sub(attention.apply(input) as tf.Tensor).sum().dataSync()[0]).toBeLessThan(1e-6);
    });
});
