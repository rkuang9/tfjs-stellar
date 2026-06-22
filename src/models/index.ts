import { LlmModel, type LlmModelArgs } from "./llm_model";
import { GptModel, type GptModelArgs } from "../models/gpt_model";
import { createUNet, UNetModel, type UNetArgs } from "../models/u_net";
export {
    LlmModel, LlmModelArgs,
    GptModel, GptModelArgs,
    UNetModel, UNetArgs
}

export function llmModel(args: LlmModelArgs) {
    return new LlmModel(args);
}


export function gptModel(args: GptModelArgs) {
    return new GptModel(args);
}


export function unetModel(args: UNetArgs) {
    return createUNet(args);
}
