import type { Tensor } from "@tensorflow/tfjs";


export declare abstract class LazyIterator<T> {
    abstract next(): Promise<IteratorResult<T>>;
}


export declare abstract class Dataset<T> {
    abstract iterator(): Promise<LazyIterator<T>>;
    size: number;
}


export type LossOrMetricFn = (yTrue: Tensor, yPred: Tensor) => Tensor;
