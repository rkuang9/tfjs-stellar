import * as tf from "@tensorflow/tfjs";
import { categoricalCrossentropy, binaryCrossentropy } from "@tensorflow/tfjs-layers/dist/losses";

const epsilon = 1e-7;

const REDUCE_HW = [1, 2]; // reduce over width and height
const REDUCE_BHW = [0, 1, 2]; // reduce over batch, width, height
const REDUCE_BHWC = [0, 1, 2, 3]; // reduce all dimensions


// Standard (Sorensen) Dice Loss
export function diceBinaryStandard(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {

    const y_true_flat = tf.reshape(y_true, [y_true.shape[0], -1]);
    const y_pred_flat = tf.reshape(y_pred, [y_pred.shape[0], -1]);

    const intersection = tf.sum(tf.mul(y_true_flat, y_pred_flat), 1);
    const union = tf.add(tf.sum(y_true_flat, 1), tf.sum(y_pred_flat, 1));

    const dice = tf.div(
        intersection.mul(2).add(epsilon),
        union.add(epsilon)
    );

    return tf.scalar(1).sub(dice);
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(diceBinaryStandard, "name", { value: "diceBinaryStandard", configurable: false });


// https://github.com/keras-team/keras/blob/v3.3.3/keras/src/losses/losses.py#L1983-L2010
export function diceBinaryGlobal(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {
    const y_true_flat = tf.reshape(y_true, [-1]);
    const y_pred_flat = tf.reshape(y_pred, [-1]);

    const intersection = tf.sum(tf.mul(y_true_flat, y_pred_flat));
    const union = tf.add(tf.sum(y_true_flat), tf.sum(y_pred_flat));

    const dice = tf.div(
        intersection.mul(2).add(epsilon),
        union.add(epsilon)
    );

    return tf.scalar(1).sub(dice);
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(diceBinaryGlobal, "name", { value: "diceBinaryGlobal", configurable: false });


export function diceCategoricalStandard(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {
    const intersection = tf.sum(tf.mul(y_true, y_pred), REDUCE_HW);
    const union = tf.add(y_true, y_pred).sum(REDUCE_HW);

    const dice = tf.div(
        intersection.mul(2).add(epsilon),
        union.add(epsilon)
    );

    return tf.scalar(1).sub(tf.mean(dice, -1));
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(diceCategoricalStandard, "name", { value: "diceCategoricalStandard", configurable: false });


export function diceCategoricalGeneralized(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {

    // this is done twice so we calculate it once
    const y_true_sum = y_true.sum(REDUCE_BHW);

    const weighting = tf.div(1, y_true_sum.square().add(epsilon));

    const intersection = tf.sum(tf.mul(y_true, y_pred), REDUCE_BHW).mul(weighting).sum();
    const union = tf.add(y_true_sum, y_pred.sum(REDUCE_BHW)).mul(weighting).sum();

    const dice = tf.div(
        intersection.mul(2).add(epsilon),
        union.add(epsilon)
    );

    return tf.scalar(1).sub(dice);
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(diceCategoricalGeneralized, "name", { value: "diceCategoricalGeneralized", configurable: false });


export function diceCategoricalGlobal(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {

    const intersection = tf.sum(tf.mul(y_true, y_pred), REDUCE_BHWC);
    const union = tf.add(tf.sum(y_true, REDUCE_BHWC), tf.sum(y_pred, REDUCE_BHWC));

    const dice = tf.div(
        intersection.mul(2).add(epsilon),
        union.add(epsilon)
    );

    return tf.scalar(1).sub(dice);
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(diceCategoricalGlobal, "name", { value: "diceCategoricalGlobal", configurable: false });


/**
 * Calculates the Sorensen-Dice coefficient and the binary cross entropy losses.
 * Both have equal weight.
 * 
 * @param y_true the label tensor
 * @param y_pred the prediction tensor (not sparse)
 * @returns a tensor of shape `[ batch ]`
 */
export function diceBinaryCrossentropy(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {
    // reduce cross entropy shape from [B, H, W] to [B] to match dice
    const bce = binaryCrossentropy(y_true, y_pred).mean(REDUCE_HW);
    const dice = diceBinaryStandard(y_true, y_pred);

    return tf.add(bce.mul(0.5), dice.mul(0.5));
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(diceBinaryCrossentropy, "name", { value: "diceBinaryCrossentropy", configurable: false });


/**
 * Calculates the Sorensen-Dice coefficient and the categorical cross entropy losses.
 * Both have equal weight. Expects dense (non-sparse) label tensors.
 * 
 * This does not support sparse tensors because TFJS's
 * sparseCategoricalCrossentropy loss onehots the label
 * and calls categoricalCrossentropy. See
 * https://github.com/tensorflow/tfjs/blob/0fc04d958ea592f3b8db79a8b3b497b5c8904097/tfjs-layers/src/losses.ts#L143-L146
 * 
 * @param y_true the label  
 * @param y_pred the prediction tensor (not sparse)
 * @returns a tensor of shape `[ batch ]`
 */
export function diceCategoricalCrossentropy(y_true: tf.Tensor, y_pred: tf.Tensor): tf.Tensor {
    // reduce cross entropy shape from [B, H, W] to [B] to match dice
    const cce = categoricalCrossentropy(y_true, y_pred).mean(REDUCE_HW);
    const dice = diceCategoricalStandard(y_true, y_pred);

    return tf.add(cce.mul(0.5), dice.mul(0.5));
}


// prevents minification of function name which TFJS relies on
Object.defineProperty(diceCategoricalCrossentropy, "name", { value: "diceCategoricalCrossentropy", configurable: false });
