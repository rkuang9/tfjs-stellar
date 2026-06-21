import * as tf from "@tensorflow/tfjs";
import * as tfc from "@/index";
import { sparseCategoricalCrossentropy } from "@tensorflow/tfjs-layers/dist/losses";
import { Dataset, type LossOrMetricFn } from "@/tfjs_types";
import { causal as generateCausalMask } from "@/masks";
import { KvCacheContainer } from "@/kv_cache";


// eslint-disable-next-line
export interface LlmModelArgs extends tf.SequentialArgs {
};


interface DatasetArgs extends tf.TensorContainerObject {
    xs: tf.Tensor;
    ys: tf.Tensor;
    loss_mask?: tf.Tensor;
    packing_mask?: tf.Tensor;
}


/**
 * This class overrides the `fitDataset()` function of tf.Sequential to support loss
 * and packing masking. Use the `generate()` function to autoregressively predict the
 * next, set `stopPredicting=true` to stop.
 */
export class LlmModel extends tf.Sequential {
    static className = "LlmModel";

    private stopPredicting_: boolean = true;

    constructor(args: LlmModelArgs) {
        args.name = args.name ?? "model";
        super(args);
    }


    /**
     * Returns the metric functions and names so that metrics can be reported
     * as they are in the base version of model.fitDataset
     * 
     * e.g. "categoricalAccuracy" should be reported as "acc"
     */
    protected getMetricFunctions() {
        const [loss, ...metric_fn_names] = this.metricsNames;

        return this.metricsTensors.map((metric_tensor, index) => ({
            metric_fn: metric_tensor[0],
            metric_label: metric_fn_names[index]
        }))
    }


    /**
     * Get exactly one loss function from the loss function provided in `model.compile()`.
     * If a string identifier was used, convert it to the actual loss function.
     */
    protected getLossFunction(): LossOrMetricFn {
        let loss = this.loss;

        if (Array.isArray(loss)) {
            loss = loss[0];
        }

        if (typeof loss == "string") {
            if (loss == "sparseCategoricalCrossentropy") {
                return sparseCategoricalCrossentropy;
                /* throw Error("LlmModel.getLossFunction: TFJS's sparseCategoricalCrossentropy" +
                    " is not truly sparse, it simply converts it to onehot." +
                    " Use categoricalCrossentropy instead. See" +
                    " https://github.com/tensorflow/tfjs/blob/0fc04d958ea592f3b8db79a8b3b497b5c8904097/tfjs-layers/src/losses.ts#L143-L146"); */
            }

            const loss_id = loss as string;

            const loss_fn =
                ((tfc.losses as Record<string, any>)[loss_id] ??
                    (tf.losses as Record<string, any>)[loss_id] ??
                    (tf.metrics as Record<string, any>)[loss_id]) as LossOrMetricFn

            if (loss_fn) {
                return loss_fn
            } else {
                throw Error(`LlmModel.getLossFunction: ${loss_id} is not a valid loss function`);
            }
        } else if (typeof loss == "function") {
            return loss;
        }

        throw Error("LlmModel.getLossFunction: the loss function's type should be string or function");
    }


    /**
     * Train on a `tf.data.generator` dataset. See https://js.tensorflow.org/api/latest/#data.generator.
     * 
     * The generator should yield `xs`, `ys`, `loss_mask` (if fine-tuning), and
     * `packing_mask` (if sequence packing was done)
     * 
     * @param tfdataset an instance of a `tf.Dataset` generator
     * @param args a ModelFitDatasetArgs
     */
    override async fitDataset<T = DatasetArgs>(tfdataset: Dataset<T>, args: tf.ModelFitDatasetArgs<T>): Promise<any> {
        this.stopTraining = false;

        const dataset = tfdataset as tf.data.Dataset<DatasetArgs>;
        const { epochs, callbacks } = args;

        const metric_functions = this.getMetricFunctions();
        const loss_function = this.getLossFunction();
        this.lossFunctions = [loss_function];

        const {
            onBatchBegin,
            onBatchEnd,
            onEpochBegin,
            onEpochEnd,
            onTrainBegin,
            onTrainEnd,
        } = callbacks as tf.CustomCallbackArgs ?? {};

        await onTrainBegin?.();

        let cached_causal_mask: tf.Tensor | undefined = undefined;

        for (let epoch = 0; epoch < epochs; epoch++) {
            await onEpochBegin?.(epoch);

            let batch = 0;
            let total_samples = 0;
            const accumulated_epoch_metrics: { [metric: string]: number } = {};

            // loop through dataset using its iterator
            const iterator = await dataset.iterator();
            let sample = await iterator.next();

            while (!sample.done) {
                const batch_metrics: { [metric: string]: number } = { batch };

                const { xs, ys, loss_mask, packing_mask } = sample.value;
                const batch_size = xs.shape[0];
                total_samples += batch_size; // for epoch metrics averaging

                if (xs.shape.length != 2) {
                    throw Error(`LlmModel.fitDataset: ${this.name} the generator dataset should be batched, run: dataset.batch(batch_size)`);
                }

                // pre-calculate the causal attention mask and reuse it for all attention layers,
                const seq_length = xs.shape[xs.shape.length - 1];

                if (!cached_causal_mask || cached_causal_mask.shape[0] != seq_length) {
                    cached_causal_mask = generateCausalMask(seq_length, seq_length);
                }

                await onBatchBegin?.(batch);

                tf.tidy(() => {
                    const { y_pred, loss } = this.fitBatch(xs, ys, loss_mask, loss_function, {
                        packingMask: packing_mask,
                        causalMask: cached_causal_mask
                    })

                    const loss_value = (loss.dataSync())[0];

                    batch_metrics.loss = loss_value;
                    accumulated_epoch_metrics.loss = (accumulated_epoch_metrics.loss || 0) + loss_value * batch_size;

                    // calculate and store metrics
                    for (const { metric_fn, metric_label } of metric_functions) {
                        const metric_sum = metric_fn(ys, y_pred!).mean();

                        const metric_value = (metric_sum.dataSync())[0];

                        batch_metrics[metric_label] = metric_value// / batch_size;
                        accumulated_epoch_metrics[metric_label] = (accumulated_epoch_metrics[metric_label] || 0) + metric_value * batch_size;
                    }

                    tf.dispose(y_pred!);
                })

                tf.dispose(xs);
                tf.dispose(ys);
                tf.dispose(loss_mask);

                if (packing_mask) {
                    tf.dispose(packing_mask);
                }

                await onBatchEnd?.(batch, batch_metrics);

                // so that stop training works
                await tf.nextFrame();

                if (this.stopTraining) {
                    break;
                }

                sample = await iterator.next();
                batch++;
            }

            for (const metric in accumulated_epoch_metrics) {
                accumulated_epoch_metrics[metric] = accumulated_epoch_metrics[metric] / total_samples;
            }

            await onEpochEnd?.(epoch, accumulated_epoch_metrics);

            if (this.stopTraining) {
                break;
            }
        }

        tf.dispose(cached_causal_mask);
        await onTrainEnd?.()

        return {};
    }


    /**
     * Run the core forward and backward propagation on one training batch. This
     * should be called within a `tf.tidy()`.
     * 
     * @param xs the sample/input tensor
     * @param ys the label/target tensor
     * @param loss_mask a loss mask to ignore the prediction's non-assistant tokens
     * @param loss_function the model's loss function
     * @param other_masks other masks used by the model's layers e.g. packing mask, causal mask
     */
    protected fitBatch(
        xs: tf.Tensor,
        ys: tf.Tensor,
        loss_mask: tf.Tensor | undefined,
        loss_function: LossOrMetricFn,
        other_masks?: { [key: string]: tf.Tensor | undefined }
    ): {
        y_pred: tf.Tensor<tf.Rank>;
        loss: tf.Scalar;
    } {
        let y_pred: tf.Tensor;

        // forward pass, calculate loss
        const { value: loss, grads } = tf.variableGrads(() => {
            // prediction has shape [batch, sequence_length, vocab_size]
            y_pred = this.apply(xs, {
                training: true,
                ...other_masks
            }) as tf.Tensor;

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


    override compile(args: tf.ModelCompileArgs): void {
        if (args.loss == "categoricalCrossentropy") {
            throw Error(`LlmModel.compile: use sparseCategoricalCrossentropy loss (along with onehot encoded labels) instead of categoricalCrossEntropy`)
        }

        super.compile(args);
    }


    /**
     * Autoregressively generate the next token until `model.stopPredicting` is set
     * to `true` or the KV cache reaches its maximum sequence length. For a single chat
     * session, the input should only be the most recent prompt(s). The KV cache stores
     * the prior chat history up until the most recent chat.
     * 
     * @param input tokenized input of the newest chat
     * @param kv_cache an instance of a KV cache container
     * @param onPredict callback function to receive the most recent token predicted
     */
    public async generate(input: tf.Tensor1D, kv_cache: KvCacheContainer, onPredict: (token: tf.Tensor) => Promise<void>) {
        if (kv_cache.size >= kv_cache.maxSequenceLength) {
            throw Error(`LlmModel.generate: ${this.name} KV cache's size reached the maxSequenceLength (${kv_cache.maxSequenceLength})`);
        }

        this.stopPredicting = false;

        let current_token: tf.Tensor2D = tf.tidy(() => input.expandDims(0)) as tf.Tensor2D; // it's 2D because of the required batch dimension

        while (!this.stopPredicting && kv_cache.size < kv_cache.maxSequenceLength) {
            // add a batch dimension because forward pass requires inputs batched
            const next_token = tf.tidy(() => this.predictNextToken(current_token, kv_cache));

            // pass back the predicted token, without the batch dim,
            const unbatched_next_token = tf.tidy(() => next_token.squeeze([0]));
            await onPredict(unbatched_next_token);

            unbatched_next_token.dispose();

            current_token.dispose();
            current_token = next_token;
        }

        tf.dispose(current_token);
    }


    /**
     * Given a tokenized sentence, predict the next token (word).
     * A normal prediction is ran to get an output with the shape
     * `[ batch_size, sentence_length, vocab_size ]` and the `vocab_size`
     * position with the highest scored probability in the last
     * position of `sentence_length` is returned as the next predicted
     * token.
     */
    public predictNextToken(input: tf.Tensor2D, kv_cache: KvCacheContainer) {
        if (input.shape[0] != 1) {
            throw Error(`LlmModel.predictNextToken: ${this.name} expects an input with a batch size of 1`);
        }

        return tf.tidy(() => {
            // comes back as [batch, sequence_length, vocab_size]
            const prediction = this.apply(input, { kvCache: kv_cache }) as tf.Tensor;

            const [batch_size, sequence_length, vocab_size] = prediction.shape;

            // get the last token
            const next_token = prediction.slice([0, sequence_length - 1, 0], [batch_size, 1, vocab_size]).argMax(2)

            return next_token as tf.Tensor2D;
        })
    }


    get stopPredicting() {
        return this.stopPredicting_;
    }


    set stopPredicting(stop: boolean) {
        this.stopPredicting_ = stop;
    }

}


tf.serialization.registerClass(LlmModel);
