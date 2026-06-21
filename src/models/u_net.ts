import * as tf from "@tensorflow/tfjs";
import { type ActivationIdentifier } from "@tensorflow/tfjs-layers/dist/keras_format/activation_config";


export interface UNetArgs {
    /**
     * The starting number of filters.
     */
    filters: number;
    /**
     * The number of categories. For binary segmentation, `units=1`.
     */
    units: number;
    /**
     * The activation of the final output convolution layer. Defaults to `sigmoid` if `categories=1`, else `softmax`.
     */
    activation?: ActivationIdentifier;
    /**
     * The depth of the U-Net or the number of contractions and the number of expansions.
     */
    depth: number;
    /**
     * Adds residual connections to transform the model into a ResUNet. Defaults to `false`.
     */
    residual?: boolean;
    /**
     * Adds batch normalization to convolutions. Best used for batched inputs. Defaults to `false`.
     */
    batchNorm?: boolean;
    /**
     * Set the unbatched input shape of the U-Net in the format `[height, width, channels]`. Defaults to `[null, null, 3]`. If set, only channels is mandatory.
     */
    inputShape?: [number | null, number | null, number];
}


export type UNetModelArgs = UNetArgs & Omit<tf.SequentialArgs, "layers">;


export class UNetModel extends tf.Sequential {

    constructor(args: UNetModelArgs) {
        const {
            filters,
            units,
            activation = units == 1 ? "sigmoid" : "softmax",
            depth,
            residual = false,
            batchNorm = false,
            inputShape = [null, null, 3],
            ...sequentialArgs
        } = args;

        sequentialArgs.name = sequentialArgs.name ?? "unet_model";

        super({
            ...sequentialArgs,
            // calling user should not modify the layers after instantiation
            layers: [createUNet({ filters, units, activation, depth, residual, batchNorm, inputShape })]
        });
    }


    override summary(lineLength?: number, positions?: number[], printFn?: (message?: any, ...optionalParams: any[]) => void): void {
        super.summary(lineLength, positions, printFn);
        (this.layers[0] as tf.LayersModel).summary(lineLength, positions, printFn);
    }
}


export function createUNet({ filters, depth, units, activation, residual = false, batchNorm = false, inputShape = [null, null, 3] }: UNetModelArgs) {
    if (units < 1) {
        throw Error(`createUNet: units should be >= 1, got ${units}`);
    }

    const [image_height, image_width] = inputShape;
    const divisble_by = 2 ** depth;

    if ((image_height != null && image_height % divisble_by != 0) ||
        image_width != null && image_width % divisble_by != 0) {
        throw Error(`createUNet: the input height and width must be divisible by 2^depth (${divisble_by})`)
    }

    const input = tf.input({ shape: inputShape });

    const skip_connection: tf.SymbolicTensor[] = [];

    let x = input;

    // calculate the filter sizes for each level
    const filter_sizes = Array.from({ length: depth }, (_, i) => filters * (2 ** i));

    for (const filter_size of filter_sizes) {
        const contraction = contractionBlock(x, filter_size, residual, batchNorm, `contraction-f${filter_size}`);

        x = tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: `pool-f${filter_size}` }).apply(contraction) as tf.SymbolicTensor;
        skip_connection.push(contraction);
    }

    x = contractionBlock(x, filter_sizes.at(-1)! * 2, residual, batchNorm, "bottleneck");

    for (let i = filter_sizes.length - 1; i >= 0; i--) {
        x = expansionBlock(x, skip_connection[i], filter_sizes[i], residual, batchNorm, `expansion-f${filter_sizes[i]}`);
    }

    const output = tf.layers.conv2d({
        filters: units,
        kernelSize: 1,
        padding: "same",
        activation: activation ?? (units == 1 ? "sigmoid" : "softmax"),
        name: "output-conv"
    }).apply(x) as tf.SymbolicTensor;

    return tf.model({ inputs: input, outputs: output, name: "u_net" });
}


export async function loadUNetModel(pathOrIOHandler: string | tf.io.IOHandler, options?: tf.io.LoadOptions) {
    const model = await tf.loadLayersModel(pathOrIOHandler, options);
    const unet = createUNet({ depth: 1, filters: 4, units: 1 }); // these are dummy args that are overwritten
    const { name, ...rest } = model;
    Object.assign(unet, rest);

    return unet;
}


/**
 * The contraction block of a U-Net
 * 
 * Conv > BN > ReLU > Conv > BN + residual > ReLU
 * 
 * TODO: for residual, change order to (BN > ReLU > Conv)x2 + residual
 * 
 * @param x a previous layer's symbolic output
 * @param filters the number of filters, usually half the previous expansion block's
 * @param residual includes a residual connection
 * @param batchNorm applies batch normalization before ReLU activation
 * @param name a unique name for the contraction block
 */
function contractionBlock(x: tf.SymbolicTensor, filters: number, residual: boolean, batchNorm: boolean, name: string) {

    const conv1 = tf.layers.conv2d({
        filters,
        kernelSize: 3,
        padding: "same",
        useBias: !batchNorm,
        kernelInitializer: "heNormal",
        name: `${name}-1-conv2d`
    });
    const relu1 = tf.layers.reLU({ name: `${name}-1-relu` });

    const conv2 = tf.layers.conv2d({
        filters,
        kernelSize: 3,
        padding: "same",
        useBias: !batchNorm,
        kernelInitializer: "heNormal",
        name: `${name}-2-conv2d`
    });
    const relu2 = tf.layers.reLU({ name: `${name}-2-relu` });

    let forward = conv1.apply(x);

    if (batchNorm) {
        forward = tf.layers.batchNormalization({ name: `${name}-1-batchnorm` }).apply(forward);
    }

    forward = relu1.apply(forward);

    forward = conv2.apply(forward);

    if (batchNorm) {
        forward = tf.layers.batchNormalization({ name: `${name}-2-batchnorm` }).apply(forward);
    }

    if (residual) {
        let residual_skip = x;

        if (x.shape[x.shape.length - 1] != filters) {
            // a 1x1 convolution on the input to ensure the residual connection's
            // channels/filters dim matches the convolution output
            residual_skip = tf.layers.conv2d({
                filters,
                kernelSize: 1,
                padding: "same",
                useBias: !batchNorm,
                kernelInitializer: "heNormal",
                name: `${name}-residual`
            }).apply(x) as tf.SymbolicTensor;
        }

        if (batchNorm) {
            residual_skip = tf.layers.batchNormalization({
                name: `${name}-residual-batchnorm`
            }).apply(residual_skip) as tf.SymbolicTensor;
        }

        forward = tf.layers.add().apply([
            residual_skip as tf.SymbolicTensor,
            forward as tf.SymbolicTensor
        ])
    }

    forward = relu2.apply(forward);

    return forward as tf.SymbolicTensor;
}


/**
 * The expansion block of a U-Net
 * 
 * Upconv + skip > contraction block
 *  
 * @param x a previous layer's symbolic output
 * @param skip the corresponding contraction block's output (before pool), shape matches `x`
 * @param filters the number of filters, usually half the previous expansion block's
 * @param residual includes a residual connection
 * @param batchNorm apply batch normalization, should be `false` when batch size is `1`
 * @param name a unique name for the contraction block
 */
function expansionBlock(x: tf.SymbolicTensor, skip: tf.SymbolicTensor, filters: number, residual: boolean, batchNorm: boolean, name: string) {

    const upconv = tf.layers.conv2dTranspose({
        filters,
        padding: "same",
        kernelSize: 2,
        strides: 2,
        kernelInitializer: "heNormal",
        name: `${name}-upconv`
    });

    const concat = tf.layers.concatenate({ axis: -1, name: `${name}-concat-upconv-skip` });

    let forward = upconv.apply(x) as tf.SymbolicTensor;
    forward = concat.apply([forward, skip]) as tf.SymbolicTensor;

    return contractionBlock(forward, filters, residual, batchNorm, name);
}
