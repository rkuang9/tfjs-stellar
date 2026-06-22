import * as tf from "@tensorflow/tfjs";


export interface KvCacheArgs {
    batchSize: number;
    maxSequenceLength: number;
    numHeads: number;
    headDim: number;
    dtype?: tf.DataType
}


export function kvCacheContainer(maxSequenceLength: number) {
    return new KvCacheContainer(maxSequenceLength);
}


export function kvCache(args: KvCacheArgs) {
    return new KvCache(args);
}


/**
 * A container for KV caches. A model should initialize one KV cache
 */
export class KvCacheContainer {
    protected caches = new Map<string, KvCache>();
    protected max_sequence_length: number;


    constructor(maxSequenceLength: number) {
        if (!maxSequenceLength) {
            throw Error(`KvCacheContainer: expected KV cache maximum sequence length to be greater than 0, got: ${String(maxSequenceLength)}`);
        }

        this.max_sequence_length = maxSequenceLength;
    }


    public create(id: string, args: Omit<KvCacheArgs, "maxSequenceLength">) {
        const new_cache = new KvCache({
            ...args,
            maxSequenceLength: this.max_sequence_length
        });

        this.caches.set(id, new_cache);
    }


    /**
     * The key and value tensors should have the shape (post head split, etc) `[batch, heads, seq, head_dim]`
     */
    public update(id: string, key: tf.Tensor4D, value: tf.Tensor4D) {
        const kv_cache = this.caches.get(id);

        if (!kv_cache) {
            return undefined;
        }

        const { keyCache, valueCache } = kv_cache.update(key, value);

        // slicing to get only the past key and value projections, but normally
        // in TensorFlow and PyTorch the full cache is returned and masked for
        // graph purposes
        return tf.tidy(() => {
            const k_cache = keyCache.slice(
                [0, 0, 0, 0],
                [keyCache.shape[0], keyCache.shape[1], kv_cache.size, keyCache.shape[3]]);
            const v_cache = valueCache.slice(
                [0, 0, 0, 0],
                [valueCache.shape[0], valueCache.shape[1], kv_cache.size, valueCache.shape[3]]);

            return {
                keyCache: k_cache,
                valueCache: v_cache
            }
        })
    }


    public reset() {
        this.caches.forEach(cache => {
            cache.reset();
        })
    }


    public dispose() {
        this.caches.forEach(cache => {
            cache.dispose();
        })
    }


    public get size() {
        // the size of all KV caches are expected to be the same, just use the first one
        return this.caches.entries().next().value?.[1].size ?? 0;
    }


    public get maxSequenceLength() {
        return this.max_sequence_length;
    }
}


export class KvCache {

    protected key_cache: tf.Variable<tf.Rank.R4>;
    protected value_cache: tf.Variable<tf.Rank.R4>

    // the size of the KV cache, represents the number of tokens since the first chat token
    protected current_position: number = 0;

    protected batch_size: number;
    protected max_sequence_length: number;
    protected num_kv_heads: number;
    protected head_dim: number;

    constructor({ batchSize, maxSequenceLength, numHeads, headDim, dtype = "float32" }: KvCacheArgs) {
        const cache_shape = [batchSize, numHeads, maxSequenceLength, headDim] as [number, number, number, number];

        this.key_cache = tf.variable(tf.zeros(cache_shape, dtype), false);
        this.value_cache = tf.variable(tf.zeros(cache_shape, dtype), false);

        this.batch_size = batchSize;
        this.max_sequence_length = maxSequenceLength;
        this.num_kv_heads = numHeads;
        this.head_dim = headDim;
    }


    /**
     * The key and value tensors should have the shape (post head split, etc) `[batch, heads, seq, head_dim]`
     */
    public update(key: tf.Tensor4D, value: tf.Tensor4D) {
        const batch_size = key.shape[0];
        const seq_len = key.shape[2];

        if (batch_size > this.key_cache.shape[0]) {
            throw Error(`The current KV cache has been set up with a batch size of` +
                ` ${this.key_cache.shape[0]}, but found new key tensors with batch size ${batch_size}`)
        }

        if (this.current_position + seq_len > this.max_sequence_length) {
            throw Error(`The KV cache has exceeded its maximum sequence length of ${this.max_sequence_length}. Use a larger value.`);
        }

        const new_key_cache = this.mergeIntoCache(key, this.key_cache);
        const new_value_cache = this.mergeIntoCache(value, this.value_cache);

        this.key_cache.assign(new_key_cache);
        this.value_cache.assign(new_value_cache);

        new_key_cache.dispose();
        new_value_cache.dispose();

        // advance the pointer to reflect the updated cache's current
        this.current_position += seq_len;

        return {
            keyCache: this.key_cache,
            valueCache: this.value_cache,
        }
    }


    protected mergeIntoCache(new_value: tf.Tensor4D, current_cache: tf.Tensor4D) {
        const seq_len = new_value.shape[2];

        return tf.tidy(() => {

            const historical = current_cache.slice(
                [0, 0, 0, 0],
                [this.batch_size, this.num_kv_heads, this.current_position, this.head_dim]);

            const future = current_cache.slice(
                [0, 0, this.current_position + seq_len, 0],
                [this.batch_size, this.num_kv_heads, this.max_sequence_length - this.current_position - seq_len, this.head_dim]);

            // merge the new tensor into the current cache to create a new, larger, cache,
            // this is different from Python immplementations because TFJS tensors are immutable,
            // because we cannot update a slice, we must slice and concat
            return tf.concat([historical, new_value, future], 2);
        })
    }


    public reset(): void {
        this.current_position = 0;

        tf.tidy(() => {
            const key_cache_shape = this.key_cache.shape;
            const value_cache_shape = this.value_cache.shape;

            this.key_cache.assign(tf.zeros(key_cache_shape));
            this.value_cache.assign(tf.zeros(value_cache_shape));
        });
    }


    public dispose(): void {
        this.key_cache.dispose();
        this.value_cache.dispose();
    }


    /**
     * The size of the KV cache, also the number of tokens since the first one.
     */
    get size(): number {
        return this.current_position;
    }

}
