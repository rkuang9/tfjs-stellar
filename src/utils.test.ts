import * as tf from "@tensorflow/tfjs";
import { getScaleShape, getRandomCropStart } from "@/utils";
import { causal } from "@/masks";

// avoid TFJS node message during Jest testing
tf.env().set('IS_NODE', false);


describe("test custom TFJS utility functions", () => {

    test("crop an image using the same shape, results in same shape", async () => {
        // cropping an image of the same shape
        const img_size = [133, 84] as [number, number];
        const target_size = [133, 84] as [number, number];

        expect(getRandomCropStart(img_size, target_size)).toEqual([0, 0, 0]);
    });


    it("should throw when crop is larger than image", async () => {
        expect(() => getRandomCropStart([128, 128], [1000, 2000])).toThrow();
    })


    test("cropped image shape", async () => {
        // cropping from wide to tall image
        for (let i = 0; i < 100; i++) {
            const img_size = [4923, 832] as [number, number];
            const target_size = [333, 739] as [number, number];

            const [crop_start_h, crop_start_w, channels] = getRandomCropStart(img_size, target_size)

            expect(crop_start_h).toBeLessThanOrEqual(img_size[0] - target_size[0]);
            expect(crop_start_w).toBeLessThanOrEqual(img_size[1] - target_size[1]);
        }

        // cropping from tall to wide image
        for (let i = 0; i < 100; i++) {
            const img_size = [381, 999] as [number, number];
            const target_size = [300, 157] as [number, number];

            const [crop_start_h, crop_start_w, channels] = getRandomCropStart(img_size, target_size)

            expect(crop_start_h).toBeLessThanOrEqual(img_size[0] - target_size[0]);
            expect(crop_start_w).toBeLessThanOrEqual(img_size[1] - target_size[1]);
        }
    });


    test("scale 1:1, results in the same shape", async () => {
        const scale = getScaleShape([256, 256], [256, 256])
        expect(scale).toEqual([256, 256]);
    });


    test("scaled image shape", async () => {
        // scaling squares result in squares
        const scale1 = getScaleShape([256, 256], [128, 128])
        expect(scale1).toEqual([128, 128]);

        const scale2 = getScaleShape([128, 128], [256, 256])
        expect(scale2).toEqual([256, 256]);

        const scale3 = getScaleShape([123, 123], [321, 321])
        expect(scale3).toEqual([321, 321]);

        const scale4 = getScaleShape([321, 321], [123, 123])
        expect(scale4).toEqual([123, 123]);

        // scaling rectangles result in rectangles
        const scale5 = getScaleShape([640, 480], [1280, 960])
        expect(scale5).toEqual([1280, 960]);

        const scale6 = getScaleShape([480, 640], [960, 1280])
        expect(scale6).toEqual([960, 1280]);

        const [scale7_h, scale7_w] = getScaleShape([777, 555], [555, 333])
        expect(scale7_h).toBeGreaterThan(scale7_w);

        const [scale8_h, scale8_w] = getScaleShape([555, 777], [333, 555])
        expect(scale8_h).toBeLessThan(scale8_w);
    });


    test("causal attention map", async () => {
        const seq_len = 4;
        const causal_mask = causal(seq_len, seq_len);

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
    });

});
