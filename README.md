# tfjs-stellar
An extension of TensorFlow.js for implementing large language models.

# Install

```bash
npm install @stellarapp/tfjs-stellar
```

# Layers
- MultiHeadAttention
- CachedRopeMultiHeadAttention
- TransformerDecoder
- TransformerEncoder
- GPT2DecoderBlock
- RotaryPositionEmbedding
- PositionalEncoding
- TokenAndPositionalEmbedding

> **Warning**:
> These layers are not compatible with their TensorFlow Keras equivalent.


## Models
- LlmModel
- GptModel
- KvCache
- UNetModel


## Masks
- Causal
- Packing


## Example

```ts
import * as tfs from "@stellarapp/tfjs-stellar";
import * as tf from "@tensorflow/tfjs";

const attention = tfs.layers.multiheadAttention({ numHeads: 1, embedDim: 64 });
const output = attention.apply(tf.randomUniform([1, 5, 64]));

const gpt_model = tfs.models.gptModel({ numLayers: 1, numHeads: 1, embedDim: 64, vocabSize: 128 });
gpt_model.compile({ loss: "sparseCategoricalCrossentropy", optimizer: "adam" });
gpt_model.summary();

// see https://js.tensorflow.org/api/latest/#data.generator
// on how to create a generator dataset
//gpt_model.fitDataset(your_generator_dataset, { epochs: 1 });
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