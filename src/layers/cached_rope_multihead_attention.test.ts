import * as tf from '@tensorflow/tfjs';

import { KvCacheContainer } from '@/kv_cache';
import { CachedRoPEMultiHeadAttention } from '@/layers/cached_rope_multihead_attention';


// disables warning for using the faster node backend,
// https://github.com/tensorflow/tfjs/issues/5349#issuecomment-885170504
tf.env().set('IS_NODE', false);


describe("CachedRoPEMultiHeadAttention tests", () => {

    test("aggregate forward passes output are identical normal multihead attention", () => {
        compareNormalWithCachedAttention(tf.randomUniform<tf.Rank.R3>([2, 10, 16]), 123);
        compareNormalWithCachedAttention(tf.randomUniform<tf.Rank.R3>([1, 10, 16]), 123);
        compareNormalWithCachedAttention(tf.randomUniform<tf.Rank.R3>([1, 1, 16]), 123);
        compareNormalWithCachedAttention(tf.randomUniform<tf.Rank.R3>([3, 2, 16]), 123);

        // input exceeds KV cach size
        expect(() => compareNormalWithCachedAttention(tf.randomUniform<tf.Rank.R3>([1, 10, 16]), 5)).toThrow();

        function compareNormalWithCachedAttention(input: tf.Tensor3D, max_sequence_length: number) {
            const embed_dim = input.shape[2];
            const batch = input.shape[0];
            const heads = 2;

            const kv_cache = new KvCacheContainer(max_sequence_length);

            const normal_mha = new CachedRoPEMultiHeadAttention({ numHeads: heads, embedDim: embed_dim, causal: true });
            const normal_mha_output = normal_mha.apply(input) as tf.Tensor;

            // initialize cached attention with identical configuration and weights
            const cached_mha1 = new CachedRoPEMultiHeadAttention({ ...normal_mha.getConfig(), name: "cache_test1" });
            cached_mha1.build(input.shape);
            cached_mha1.setWeights(normal_mha.getWeights());

            const cached_mha2 = new CachedRoPEMultiHeadAttention({ ...normal_mha.getConfig(), name: "cache_test2" });
            cached_mha2.build(input.shape);
            cached_mha2.setWeights(normal_mha.getWeights());

            const cached_mha_outputs1: tf.Tensor[] = [];
            const cached_mha_outputs2: tf.Tensor[] = [];

            for (let i = 0; i < input.shape[1]; i++) {
                const current_token = input.slice([0, i, 0], [batch, 1, embed_dim]);

                cached_mha_outputs1.push(cached_mha1.apply(current_token, { kvCache: kv_cache }) as tf.Tensor);
                cached_mha_outputs2.push(cached_mha2.apply(current_token, { kvCache: kv_cache }) as tf.Tensor);
            }

            expect(kv_cache.size== input.shape[1]);
            expect(kv_cache.size == input.shape[1]);

            expect(normal_mha_output.sub(tf.concat(cached_mha_outputs1, 1)).sum().dataSync()[0]).toBeLessThan(1e-6);
            expect(normal_mha_output.sub(tf.concat(cached_mha_outputs2, 1)).sum().dataSync()[0]).toBeLessThan(1e-6);
        }
    })
});
