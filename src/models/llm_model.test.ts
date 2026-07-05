import * as tf from "@tensorflow/tfjs";

import * as models from "../models";

// disables warning for using the faster node backend,
// https://github.com/tensorflow/tfjs/issues/5349#issuecomment-885170504
tf.env().set('IS_NODE', false);


describe("LlmModel tests", () => {
    it("should accept the string perplexity as a metric", async () => {
        const input = tf.randomUniform([1, 4]);
        const target = tf.randomUniform([1, 4, 1], 0, 1, "int32").asType("float32");

        const model = new models.GptModel({
            numHeads: 1,
            numLayers: 1,
            embedDim: 4,
            vocabSize: 128
        });

        model.compile({
            optimizer: "adam",
            loss: "sparseCategoricalCrossentropy",
            metrics: ["perplexity"]
        });

        const history = await model.fit(input, target, { epochs: 1 });
        expect(Array.isArray(history.history.perplexity) && history.history.perplexity.length > 0).toBe(true);
    })
})
