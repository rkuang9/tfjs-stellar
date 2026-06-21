import * as tf from "@tensorflow/tfjs";
import { type LossOrMetricFn } from "../tfjs_types";
import { LlmModel, type LlmModelArgs } from "../models/llm_model";
import { KvCacheContainer } from "../kv_cache";
import { type DisposeResult } from "@tensorflow/tfjs-layers/dist/engine/topology";
import { GPT2DecoderBlock } from "../layers/gpt_decoder_block";


export interface GptModelArgs extends LlmModelArgs {
    /**
     * Number of heads per attention layer.
     */
    numHeads: number;
    /**
     * Number of GPT decoder blocks.
     */
    numLayers: number;
    /**
     * The embedding size of each token.
     */
    embedDim: number;
    /**
     * The vocabulary size of the embedding layer and number of units of the output
     * layer. This is also the tokenizer vocabulary size.
     */
    vocabSize: number;
    /**
     * Pad the embeddings' vocab size and output layer's units to the next nearest 
     * multiple of 64 to optimize hardware efficiency. Defaults to `true`.
     * 
     * For example: if a tokenizer has 50,257 tokens, the model uses 50,304 for the
     * vocab size and output units count.
     */
    padToMultipleOf64?: boolean;
}


/**
 * This is a subclass of tf.Sequential that creating a GPT-like model and
 * automatically handles padding (and masking) the vocab size for hardware
 * efficiency.
 * 
 * Example:
 * 
 * ```javascript
 * 
 * const model = new GptModel({ numLayers: 1, numHeads: 1, embedDim: 16, vocabSize: 64 });
 * model.compile({ loss: "sparseCategoricalCrossentropy", optimizer: "adam" });
 * 
 * // use fitDataset() instead of fit for masking support
 * model.fitDataset(your_batched_generator_dataset, { epochs: 1 });
 * 
 * const kv_cache = new KvCacheContainer(your_preferred_max_sequence_length);
 * 
 * // use generate() and predictNextToken() instead of predict() for masking and auto memory cleanup
 * model.generate(tokenized_tensor1d_input, kv_cache, onPredict_callback)
 * 
 * 
 * ```
 */
export class GptModel extends LlmModel {
    static className = "GptModel";

    protected readonly numHeads: number;
    protected readonly numLayers: number;
    protected readonly embedDim: number;
    protected readonly vocabSize: number;
    protected readonly padToMultipleOf64: boolean;

    // this is kept for reproducibility and model history but is not important since
    // it can be calculated mathematically
    protected readonly vocabSizePadded: number;

    // the amount to pad the embedding vocab size and dense output units count
    protected vocab_padding_mask?: tf.Tensor1D;


    /**
     * DO NOT add layers in the constructor or it will break tf.loadLayersModel().
     * It should be done in build() instead.
     */
    constructor(args: GptModelArgs) {
        const { numHeads, numLayers, embedDim, vocabSize, padToMultipleOf64 = true, ...rest } = args;

        super({ name: "model", ...rest });

        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.embedDim = embedDim;
        this.vocabSize = vocabSize;
        this.padToMultipleOf64 = padToMultipleOf64;
        this.vocabSizePadded = this.padToMultipleOf64
            ? Math.ceil(this.vocabSize / 64) * 64
            : this.vocabSize;
    }


    protected override fitBatch(
        xs: tf.Tensor,
        ys: tf.Tensor,
        loss_mask: tf.Tensor | undefined,
        loss_function: LossOrMetricFn,
        other_masks?: { [key: string]: tf.Tensor | undefined }
    ) {
        let y_pred: tf.Tensor;

        // forward pass, calculate loss
        const { value: loss, grads } = tf.variableGrads(() => {
            y_pred = this.apply(xs, {
                training: true,
                ...other_masks
            }) as tf.Tensor;

            // apply vocab pad masking
            if (this.vocab_padding_mask) {
                y_pred = y_pred.add(this.vocab_padding_mask);
            }

            y_pred = tf.softmax(y_pred);

            // manually dispose later instead of the built-in disposal from variableGrads
            tf.keep(y_pred);

            const loss = loss_mask
                ? loss_function(ys, y_pred).mul(loss_mask)
                : loss_function(ys, y_pred);

            return loss.mean() as tf.Scalar;
        });

        // backpropagation
        this.optimizer.applyGradients(grads);

        return {
            y_pred: y_pred!,
            loss
        };
    }


    /**
     * Overrides LlmModel.predictNextToken to add softmax before argMax because the final
     * dense layer doesn't have an activation.
     * 
     * TODO: implement temperature and multinomial sampling so that the model has varied outputs
     */
    override predictNextToken(input: tf.Tensor2D, kv_cache: KvCacheContainer): tf.Tensor2D {
        if (input.shape[0] != 1) {
            throw Error(`GptModel.predictNextToken: ${this.name} expects an input with a batch size of 1`);
        }

        return tf.tidy(() => {
            // comes back as [batch, sequence_length, vocab_size]
            const prediction = this.apply(input, { kvCache: kv_cache }) as tf.Tensor;

            const [batch_size, sequence_length, vocab_size] = prediction.shape;

            // get the last token
            const next_token = this.vocab_padding_mask != undefined
                ? prediction
                    .slice([0, sequence_length - 1, 0], [batch_size, 1, vocab_size])
                    .add(this.vocab_padding_mask)
                    .softmax()
                    .argMax(2)
                : prediction
                    .slice([0, sequence_length - 1, 0], [batch_size, 1, vocab_size])
                    .softmax()
                    .argMax(2);

            return next_token as tf.Tensor2D;
        })
    }


    override build(inputShape?: tf.Shape | tf.Shape[]): void {
        const actual_vocab_size = this.vocabSizePadded
            ? this.vocabSizePadded
            : this.padToMultipleOf64
                ? Math.ceil(this.vocabSize / 64) * 64
                : this.vocabSize

        if (this.layers.length == 0) {
            [
                tf.layers.embedding({ inputDim: actual_vocab_size, outputDim: this.embedDim, batchInputShape: [null, null] }),
                ...Array(this.numLayers).fill(0).map(_ => new GPT2DecoderBlock({ numHeads: this.numHeads, embedDim: this.embedDim })),
                tf.layers.dense({ units: actual_vocab_size })
            ].forEach(layer => this.add(layer))
        }

        if (this.vocab_padding_mask) {
            this.vocab_padding_mask.dispose();
        }

        if (this.padToMultipleOf64 && actual_vocab_size > this.vocabSize) {
            this.vocab_padding_mask = tf.tidy(() => tf.where<tf.Tensor1D>(
                // Create a mask of padded vocab length, values after the index "vocabSize"
                // are set to -1e7 to mask out those positions so that softmax will ignore
                // them. This mask is added to the final dense layer's output
                tf.range(0, actual_vocab_size).greaterEqual(this.vocabSize), -1e7, 0).toFloat())
        }

        super.build(inputShape);
    }


    override dispose(): DisposeResult {
        this.vocab_padding_mask?.dispose();
        return super.dispose();
    }


    override getConfig() {
        const base_config = super.getConfig();

        const config = {
            numHeads: this.numHeads,
            numLayers: this.numLayers,
            embedDim: this.embedDim,
            vocabSize: this.vocabSize,
            vocabSizePadded: this.vocabSizePadded,
            padToMultipleOf64: this.padToMultipleOf64
        }

        Object.assign(config, base_config);

        return config;
    }

}


tf.serialization.registerClass(GptModel);
