# tfjs-stellar
An extension of TensorFlow.js for implementing large language models. This library can be used along side TFJS and follows its namespacing.

Questions? Want to contribute? Join the [Discord server](https://discord.gg/AUmpBEvGBw).

# Install

```bash
npm install @stellarapp/tfjs-stellar
```
This library expects the latest version of TensorFlow.js.

# Layers
- `tfs.layers.multiHeadAttention`
- `tfs.layers.cachedRopeMultiHeadAttention`
- `tfs.layers.transformerDecoder`
- `tfs.layers.transformerEncoder`
- `tfs.layers.gpt2DecoderBlock` (transformer decoder but layer norm first)
- `tfs.layers.rotaryPositionEmbedding`
- `tfs.layers.positionalEncoding`
- `tfs.layers.tokenAndPositionalEmbedding`

> **Warning**:
> These layers are not compatible with their TensorFlow Keras equivalent.


## Models
- `tfs.models.llmModel`
- `tfs.models.gptModel`
- `tfs.models.unetModel`
- `tfs.kvCacheContainer`


## Metrics

- `tfs.metrics.perplexity`
- `tfs.metrics.recall` (threshold = 0.5)
- `tfs.metrics.precision` (threshold = 0.5)

> **Warning**:
> These metrics are serializeable only with the `llmModel` and `gptModel` classes. See `llmModel.compile` override on how it's done.


## Masks
- `tfs.masks.causal`
- `tfs.masks.packing`


## Example

```ts
import * as tfs from "@stellarapp/tfjs-stellar";
import * as tf from "@tensorflow/tfjs";

// use multiheadAttention for no RoPE
const attention = tfs.layers.cachedRopeMultiheadAttention({ numHeads: 1, embedDim: 64 });
const output = attention.apply(tf.randomUniform([1, 5, 64]));



const gpt_model = tfs.models.gptModel({ numLayers: 1, numHeads: 1, embedDim: 64, vocabSize: 128 });
gpt_model.compile({ loss: "sparseCategoricalCrossentropy", optimizer: "adam", metrics: ["acc", "perplexity"] });
gpt_model.summary();

// see https://js.tensorflow.org/api/latest/#data.generator
// on how to create a generator dataset,
// only fitDataset (and not fit) supports packing and loss masking
await gpt_model.fitDataset(your_generator_dataset, { epochs: 1 });

const kv_cache = tfs.kvCacheContainer(1024);

// generate tokens until the KV cache is full or
// you set gpt_model.stopPredicting = true
gpt_model.generate(input, kv_cache, (token: tf.Tensor) => {
    // do work with the generated token, the model will wait
    // for this callback to finish before generating the next token
})

```

## Jest Unit Testing

If you plan to use this library in your Jest unit tests, you may need to add or update the following configurations in your `jest.config.ts` file's `config`

```ts
transform: {
    '^.+\\.[jt]s?$': ["ts-jest", {
        useESM: true,
    }]
},

transformIgnorePatterns: [
    "/node_modules/(?!(@stellarapp/tfjs-stellar|@tensorflow/tfjs))"
],
```