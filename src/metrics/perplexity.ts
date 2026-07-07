import * as tf from "@tensorflow/tfjs";
import { sparseCategoricalCrossentropy } from "@tensorflow/tfjs-layers/dist/losses";


export function perplexity(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {
    return tf.exp(sparseCategoricalCrossentropy(y_true, y_pred).mean());
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(perplexity, "name", { value: "perplexity", configurable: false });
