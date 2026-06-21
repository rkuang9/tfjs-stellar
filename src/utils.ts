import * as tf from "@tensorflow/tfjs";


/**
 * Calculate the desired scaled image's height and width. The shortest edge will
 * be scaled to match its corresponding target shape's edge. The longer
 * edge might end up larger than intended.
 * 
 * @param image_shape the `[height, width]` of the image
 * @param target_shape the intended `[height, width]` of the final scaled image
 */
export function getScaleShape(image_shape: tf.Shape, target_shape: [number, number]): [scaled_height: number, scaled_width: number] {
    const [img_height, img_width] = image_shape as [number, number, number];
    const [target_height, target_width] = target_shape;

    // scale based on whichever target_edge / original_edge is largest,
    // we need the following to be true (1)
    // height * scale >= target_height
    // width * scale >= target_height
    // rearranging to get an equivalent requirement (2)
    // scale >= target_height / height
    // scale >= target_width / width
    // by picking the scale value that's largest of the two, we satisfy (2), and therefore (1)
    // it may be more intuitive to think of scale as scale_h and scale_w
    const scale_factor = Math.max(target_height / img_height, target_width / img_width);
    return [Math.round(img_height * scale_factor), Math.round(img_width * scale_factor)];
}


/**
 * Calculate the starting point for a crop (slice) operation
 * on an image tensor with the shape `[height, width, channels]`.
 * 
 * @param image_shape the `[height, width]` of the image
 * @param target_shape the intended `[height, width]` of the final cropped image
 */
export function getRandomCropStart(
    image_shape: [height: number, width: number],
    target_shape: [height: number, width: number]
): [number, number, number] {
    const [img_height, img_width] = image_shape;
    const [crop_x, crop_y] = target_shape;

    if (img_height < crop_x || img_width < crop_y) {
        throw Error(`getRandomCropShape: cannot crop with a size that's bigger than,` +
            ` the image. Original [${img_height}, ${img_width}], crop [${crop_x}, ${crop_y}].`);
    }

    // there's a +1 because Math.random()'s range is [0, 1), excluding 1,
    // hence +1 to ensure the full range of possible crop starting points
    return [
        // TODO: revisit the +1
        Math.floor(Math.random() * (img_height - crop_x + 1)),
        Math.floor(Math.random() * (img_width - crop_y + 1)),
        0 // not cropping channels, so it starts at the first index
    ]
}


/**
 * Calculate the height and width padding such that the image is
 * divisible by 2^depth.
 * 
 * In U-Net image segmentation, the contraction and concatenate
 * operations requires  the input image's height and width
 * dimensions to be divisible by 2^depth.
 */
export function getPaddingForSegmentation(image: tf.Tensor3D, depth: number): [height: number, width: number] {
    const divisible = Math.pow(2, depth);

    const [height, width] = image.shape;

    return [
        (Math.ceil(height / divisible)) * divisible - height,
        (Math.ceil(width / divisible)) * divisible - width,
    ]
}


