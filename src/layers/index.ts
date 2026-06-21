import { CachedRoPEMultiHeadAttention } from "./cached_rope_multihead_attention";
import { GPT2DecoderBlock, GPTDecoderBlockArgs } from "./gpt_decoder_block";
import { MultiHeadAttention, MultiHeadAttentionArgs } from "./multihead_attention";
import { PositionalEncoding, PositionalEncodingArgs } from "./positional_encoding";
import { RotaryPositionEmbedding, RotaryPositionEmbeddingArgs } from "./rotary_position_embedding";
import { TokenAndPositionalEmbedding, TokenAndPositionalEmbeddingArgs } from "./token_and_positional_embedding";
import { TransformerDecoder, TransformerDecoderArgs } from "./transformer_decoder";
import { TransformerEncoder, TransformerEncoderArgs } from "./transformer_encoder";


export function tokenAndPositionalEmbedding(args: TokenAndPositionalEmbeddingArgs) {
    return new TokenAndPositionalEmbedding(args);
}


export function transformerEncoder(args: TransformerEncoderArgs) {
    return new TransformerEncoder(args);
}


export function transformerDecoder(args: TransformerDecoderArgs) {
    return new TransformerDecoder(args);
}


export function multiheadAttention(args: MultiHeadAttentionArgs) {
    return new MultiHeadAttention(args);
}


export function cachedRopeMultiheadAttention(args: MultiHeadAttentionArgs) {
    return new CachedRoPEMultiHeadAttention(args);
}


export function positionalEncoding(args: PositionalEncodingArgs) {
    return new PositionalEncoding(args);
}


export function gpt2DecoderBlock(args: GPTDecoderBlockArgs) {
    return new GPT2DecoderBlock(args);
}


export function rotaryPositionEmbedding(args: RotaryPositionEmbeddingArgs) {
    return new RotaryPositionEmbedding(args);
}
