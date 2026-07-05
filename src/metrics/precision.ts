import { metrics, Tensor } from "@tensorflow/tfjs";


/**
 * Applies the precision metric with the prediction rounded based on a threshold
 * 
 * @param y_true the label tensor
 * @param y_pred the prediction tensor
 * @param threshold threshold value to be considered a positive prediction, defaults to `0.5`
 * @returns 
 */
export function precision(y_true: Tensor, y_pred: Tensor, threshold: number = 0.5) {
    return metrics.precision(y_true, y_pred.greaterEqual(threshold));
}

// prevents minification of function name which TFJS relies on
Object.defineProperty(precision, "name", { value: "precision", configurable: false });
