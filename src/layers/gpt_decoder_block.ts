import * as tf from "@tensorflow/tfjs";
import { type Kwargs } from "@tensorflow/tfjs-layers/dist/types";

import { type MultiHeadAttentionArgs } from "../layers/multihead_attention";
import { TransformerDecoder, type TransformerDecoderArgs } from "../layers/transformer_decoder";


export interface GPTDecoderBlockArgs extends Omit<MultiHeadAttentionArgs, "causal"> {
    dimsFeedForward?: number;
}


/**
 * This implements the GPT-2 transformer block by modifying the transformer
 * decoder block to use pre-layer-normalization and replacing ReLU activation
 * with GELU.
 * 
 * @param numHeads number of attention heads to use
 * @param embedDim the embedding size of the input (input embeddings, typically the last dimension)
 * @param causal use causal masking on inputs (masks future inputs to prevent looking ahead), default `true`
 * @param dropout use dropout during the attention calculations, default `0.1`
 * @param dimsFeedForward the size of the intermediate feed forward layer, default `2048`
 * @param useBias use bias for the dense sublayers and multiHead attention's dense sublayers, default `true`
 */
export class GPT2DecoderBlock extends TransformerDecoder {
    static className = "GPT2DecoderBlock";


    constructor(args: TransformerDecoderArgs) {
        super(args);
    }


    /**
     * Attention sub-block which is similar to the original transformer except
     * layer normalization is applied beginning
     */
    protected override causalSelfAttentionBlock(x: tf.Tensor, kwargs: Kwargs): tf.Tensor {
        return tf.tidy(() => {
            const residual = x;

            let attention = this.causalSelfAttentionNorm.apply(x, kwargs) as tf.Tensor;
            attention = this.causalSelfAttention.apply(attention, kwargs) as tf.Tensor;
            attention = this.causalSelfAttentionDropout.apply(attention, kwargs) as tf.Tensor;
            attention = tf.add(attention, residual);

            return attention;
        });
    }


    /**
     * Feedforward sub-block which is similar to the original transformer except
     * layer normalization is applied at the beginning and gelu activation is used
     */
    protected override feedForwardBlock(x: tf.Tensor, kwargs: Kwargs): tf.Tensor {
        return tf.tidy(() => {
            const residual = x;

            let feedForward = this.feedFowardNorm.apply(x, kwargs);
            feedForward = this.feedforward1.apply(feedForward, kwargs);
            feedForward = this.feedforward2.apply(feedForward, kwargs);
            feedForward = this.feedForwardDropout.apply(feedForward, kwargs) as tf.Tensor;
            feedForward = tf.add(feedForward, residual);

            return feedForward;
        });
    }


    // the build() function does not need overriding because the layer normalization
    // outputs the same shape as its input, its position as a sub-layer doesn't affect
    // other sub-layer weight and output shapes
}


tf.serialization.registerClass(GPT2DecoderBlock);
