import * as tf from '@tensorflow/tfjs';
import { KvCacheContainer } from "../kv_cache";
import { MultiHeadAttention, type MultiHeadAttentionArgs } from '../layers/multihead_attention';
import { RotaryPositionEmbedding } from '../layers/rotary_position_embedding';
import { type Kwargs } from '@tensorflow/tfjs-layers/dist/types';


/**
 * MultiHeadAttention with RoPE and KV caching. If using KV caching, this layer
 * should be used in a custom training loop because it requires the cache to be
 * passed through the `kwargs.kvCache` argument during the `layer.apply()`
 * forward propagation.
 * 
 * If a KV cache is not provided, then this layer operates as MultiHeadAttention with RoPE.
 */
export class CachedRoPEMultiHeadAttention extends MultiHeadAttention {
    static className = "CachedRoPEMultiHeadAttention";

    protected rope: tf.layers.Layer;

    constructor(args: MultiHeadAttentionArgs) {
        super(args);
        this.rope = new RotaryPositionEmbedding({ dim: Math.floor(this.embedDim / this.numHeads) });
    }


    protected override forward(
        query_input: tf.Tensor,
        key_input: tf.Tensor,
        value_input: tf.Tensor,
        packing_mask: tf.Tensor | null,
        causal_mask: tf.Tensor | null,
        kwargs: Kwargs): tf.Tensor {

        return tf.tidy(() => {
            const { query, key, value } = this.applyInputProjections(query_input, key_input, value_input);

            // swap the seq and heads dimensions: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
            const move_head_dim_forward = [0, 2, 1, 3];

            const split = this.splitHeads(query, key, value, move_head_dim_forward);

            const query_split = split.query_split;
            let key_split = split.key_split;
            let value_split = split.value_split;

            if (kwargs.training !== true && kwargs.kvCache) {
                // runs on inference, updates the KV cache and get the historical key and value
                const cached_kv = this.getCachedKV(
                    kwargs.kvCache as KvCacheContainer, key_split, value_split);

                key_split = cached_kv.keyCache;
                value_split = cached_kv.valueCache;
            }

            // apply scaled dot production attention to get [batch, seq, numHeads, embedDim]
            const spda = MultiHeadAttention.scaledDotProductionAttention(
                query_split, key_split, value_split,
                kwargs.attentionMask ?? null, packing_mask, causal_mask,
                this.dropout, this.causal, kwargs);

            // concat heads and apply the output projection
            const output = this.outputProjection.apply(
                spda.transpose(move_head_dim_forward).reshape(
                    [query_input.shape[0], query_input.shape[1]!, this.embedDim]));

            return output as tf.Tensor;
        })
    }


    protected getCachedKV(kv_container: KvCacheContainer, key_split: tf.Tensor4D, value_split: tf.Tensor4D) {
        try {
            let kv_cache = kv_container.update(this.name, key_split, value_split);

            if (!kv_cache) {
                kv_container.create(this.name, {
                    batchSize: key_split.shape[0],
                    numHeads: this.numHeads,
                    headDim: this.embedDim / this.numHeads,
                })

                kv_cache = kv_container.update(this.name, key_split, value_split)!;
            }

            return kv_cache!;
        } catch (error: any) {
            throw Error(`${this.getClassName()}::getCachedKV ${this.name} ${error.toString()}`);
        }
    }


    /**
     * Adds RoPE position encoding right after splitting heads.
     */
    protected override splitHeads(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, shuffle: number[]) {
        const batch_size = query.shape[0];
        const split_heads = [batch_size, -1, this.numHeads, this.embedDim / this.numHeads];

        return tf.tidy(() => {
            return {
                query_split: (this.rope.apply(query.reshape(split_heads)) as tf.Tensor)
                    .transpose(shuffle) as tf.Tensor4D,
                key_split: (this.rope.apply(key.reshape(split_heads)) as tf.Tensor)
                    .transpose(shuffle) as tf.Tensor4D,
                value_split: value.reshape(split_heads).transpose(shuffle) as tf.Tensor4D
            }
        })
    }
}


tf.serialization.registerClass(CachedRoPEMultiHeadAttention);
