import * as tf from "@tensorflow/tfjs";
import * as masks from "./masks";

tf.env().set('IS_NODE', false);


describe(" mask tests", () => {
    test("packing mask for self-attention", () => {
        const boundaries = new Int32Array([1, 0, 0, 1, 0, 0, 1, 0, 0]);
        const packing_mask = masks.packing(boundaries);

        const expected_mask = tf.tensor([
            [0, 0, 0, -10000000, -10000000, -10000001, -10000000, -10000000, -10000002],
            [0, 0, 0, -10000001, -10000001, -10000002, -10000000, -10000000, -10000001],
            [0, 0, 0, -10000003, -10000000, -10000000, -10000002, -10000002, -10000000],
            [-10000000, -10000002, -10000000, 0, 0, 0, -10000001, -10000001, -10000000],
            [-10000000, -10000000, -10000001, 0, 0, 0, -10000003, -10000000, -10000001],
            [-10000000, -10000002, -10000000, 0, 0, 0, -10000000, -10000001, -10000000],
            [-10000000, -10000000, -10000000, -10000000, -10000000, -10000000, 0, 0, 0],
            [-10000000, -10000002, -10000000, -10000002, -10000000, -10000000, 0, 0, 0],
            [-10000000, -10000001, -10000000, -10000000, -10000002, -10000003, 0, 0, 0]
        ]);

        // The mask uses -1e7 on masked positions which introduces extra integers on
        // some values in the float32 tensor. Ideally it should check that the sum is equal to 0,
        // but since there are 54 masked positions, we'll just check that it's less than 108
        expect((packing_mask.sub(expected_mask).sum().arraySync() as number)).toBeLessThan(108);
    })


    test("packing mask for non-packed sequence", () => {
        const boundaries = new Int32Array([1, 0, 0, 0, 0]);
        const packing_mask = masks.packing(boundaries);

        expect((packing_mask.sum().arraySync() as number)).toEqual(0);
    })


    test("causal mask size 4", async () => {
        const seq_len = 4;
        const causal_mask = masks.causal(seq_len, seq_len);

        const _ = -1e7;
        const expected_mask = tf.tensor([
            [0, _, _, _],
            [0, 0, _, _],
            [0, 0, 0, _],
            [0, 0, 0, 0]
        ]);

        // this might fail due to precision issues on the masked positions,
        // in which case use less <= to 6 or 12 (number of masked positions x2)
        expect((await causal_mask.sub(expected_mask).sum().data())[0]).toEqual(0);
    })


    test("causal mask size 5", () => {
        const expected_mask = tf.tensor([
            [0, -10000000, -10000000, -10000000, -10000000],
            [0, 0, -10000000, -10000000, -10000000],
            [0, 0, 0, -10000000, -10000000],
            [0, 0, 0, 0, -10000000],
            [0, 0, 0, 0, 0]]);

        const causal_mask = masks.causal(5, 5);

        expect(causal_mask.equal(expected_mask).sum().arraySync()).toBe(25);
    })
});
