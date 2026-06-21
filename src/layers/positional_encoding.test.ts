import * as tf from '@tensorflow/tfjs';

import { PositionalEncoding } from '../layers/positional_encoding';

// disables warning for using the faster node backend,
// https://github.com/tensorflow/tfjs/issues/5349#issuecomment-885170504
tf.env().set('IS_NODE', false);


describe("PositionalEncoding tests", () => {
    it("should fail to instantiate a layer", () => {
        expect(() => new PositionalEncoding({ maxSequenceLength: 3, embedDim: 0 })).toThrow();
        expect(() => new PositionalEncoding({ maxSequenceLength: 3, embedDim: -1 })).toThrow();
        expect(() => new PositionalEncoding({ maxSequenceLength: 0, embedDim: 32 })).toThrow();
        expect(() => new PositionalEncoding({ maxSequenceLength: -1, embedDim: 32 })).toThrow();
    })


    test("successfull forward calls", () => {
        const embed_dims = 32;
        const sequences = 4;
        const input = tf.randomUniform([2, sequences, embed_dims]);

        const positional = new PositionalEncoding({ embedDim: embed_dims });
        expect(() => positional.apply(input)).not.toThrow();
        expect(() => positional.apply([input])).not.toThrow();
        expect(positional.computeOutputShape(input.shape)).toEqual(input.shape);
    })


    it("should throw when input sequences are too large, embedding dims don't match, input aren't rank 3", () => {
        const sequences_too_long = tf.randomUniform([100, 32]);
        const embeddings_too_large = tf.randomUniform([32, 100]);
        const wrong_rank = tf.randomUniform([10, 32, 32]);

        const positional = new PositionalEncoding({ maxSequenceLength: 10, embedDim: 32 });

        expect(() => positional.apply(sequences_too_long)).toThrow();
        expect(() => positional.apply(embeddings_too_large)).toThrow();
        expect(() => positional.apply(wrong_rank)).toThrow();
    })


    it("should return a non-empty config dict", () => {
        const attention = new PositionalEncoding({ embedDim: 32 });
        expect(Object.keys(attention.getConfig())).not.toBe(0);
    })


    // PyTorch implementation at found at
    // https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
    it("should be within 1e-6 of PyTorch's implementation", () => {
        const pytorch_embed4 = tf.tensor([
            [[0.0000000, 1.0000000, 0.0000000, 1.0000000],
            [0.8414710, 0.5403023, 0.0099998, 0.9999500],
            [0.9092974, -0.4161468, 0.0199987, 0.9998000],
            [0.1411200, -0.9899925, 0.0299955, 0.9995500],
            [-0.7568025, -0.6536436, 0.0399893, 0.9992001],
            [-0.9589243, 0.2836622, 0.0499792, 0.9987503],
            [-0.2794155, 0.9601703, 0.0599640, 0.9982005],
            [0.6569866, 0.7539023, 0.0699428, 0.9975510],
            [0.9893582, -0.1455000, 0.0799147, 0.9968017],
            [0.4121185, -0.9111302, 0.0898785, 0.9959527]]]);

        const pytorch_embed8 = tf.tensor([
            [[0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 1.0000000e+00,
                0.0000000e+00, 1.0000000e+00, 0.0000000e+00, 1.0000000e+00],
            [8.4147096e-01, 5.4030234e-01, 9.9833414e-02, 9.9500418e-01,
                9.9998331e-03, 9.9994999e-01, 9.9999981e-04, 9.9999952e-01],
            [9.0929741e-01, -4.1614684e-01, 1.9866931e-01, 9.8006660e-01,
                1.9998666e-02, 9.9980003e-01, 1.9999985e-03, 9.9999803e-01],
            [1.4112000e-01, -9.8999250e-01, 2.9552019e-01, 9.5533651e-01,
                2.9995499e-02, 9.9955004e-01, 2.9999954e-03, 9.9999553e-01],
            [-7.5680250e-01, -6.5364361e-01, 3.8941833e-01, 9.2106098e-01,
                3.9989334e-02, 9.9920011e-01, 3.9999890e-03, 9.9999201e-01],
            [-9.5892429e-01, 2.8366220e-01, 4.7942552e-01, 8.7758255e-01,
                4.9979165e-02, 9.9875027e-01, 4.9999789e-03, 9.9998754e-01],
            [-2.7941549e-01, 9.6017027e-01, 5.6464243e-01, 8.2533562e-01,
                5.9964005e-02, 9.9820054e-01, 5.9999637e-03, 9.9998200e-01],
            [6.5698659e-01, 7.5390226e-01, 6.4421761e-01, 7.6484221e-01,
                6.9942847e-02, 9.9755102e-01, 6.9999420e-03, 9.9997550e-01],
            [9.8935825e-01, -1.4550003e-01, 7.1735609e-01, 6.9670677e-01,
                7.9914689e-02, 9.9680167e-01, 7.9999138e-03, 9.9996799e-01],
            [4.1211849e-01, -9.1113025e-01, 7.8332686e-01, 6.2160999e-01,
                8.9878544e-02, 9.9595273e-01, 8.9998785e-03, 9.9995953e-01]]]);

        const positional4 = new PositionalEncoding({ embedDim: 4, maxSequenceLength: 10 });
        positional4.build([]);

        const positional8 = new PositionalEncoding({ embedDim: 8, maxSequenceLength: 10 });
        positional8.build([]);

        const margin_of_error = 1e-6;

        // the difference between this and PyTorch's implementation
        //should be insignificantly small
        expect((positional4.getWeights()[0]
            .sub(pytorch_embed4)
            .abs()
            .arraySync() as [])
            .flat(2)
            .filter(i => i > margin_of_error)
            .length).toBe(0);

        expect((positional8.getWeights()[0]
            .sub(pytorch_embed8)
            .abs()
            .arraySync() as [])
            .flat(2)
            .filter(i => i > margin_of_error)
            .length).toBe(0);
    });
});
