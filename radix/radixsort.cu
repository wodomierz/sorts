#include "../utils/cuda_device.h"
#include "../prefsum/prefsum.cuh"
#include "../utils/kernel_commons.cuh"
#include "radixsort.h"

template<int Threads, int Elements>
__device__ __forceinline__
void count_and_pref_sum_dev(int *in, int *out, int *prefixSums, int size, int mask) {
    int const BlockSize = Threads * Elements;
    __shared__ int shared[2][BlockSize];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * BlockSize;

    in += offset;
    out += offset;
    size -= offset;

    for (int thid = threadIdx.x; thid < BlockSize; thid += Threads) {
        shared[0][thid] = thid < size ? (in[thid] >> mask) & 1 : 0;
    }
    __syncthreads();

    pref_sum_and_move_result_dev<Threads, Elements>(shared, out, prefixSums, size, blockId);
}

template<int Threads, int Elements>
__device__ __forceinline__
void sort_dev(int *source, int *destination, int *prefixSum, int *prefixSums, int mask, int n, int number_of_zeros) {
    const int BlockSize = Threads * Elements;
    int blockId = one_dimension_blockId();
    int offset = blockId * BlockSize;
    prefixSum += offset;
    source += offset;
    n -= offset;
    for (int thid = threadIdx.x; thid < BlockSize && thid < n; thid += Threads) {
        int place;
        if ((source[thid] >> mask) & 1) {
            place = number_of_zeros + prefixSum[thid] + prefixSums[blockId] - 1;
        } else {
            place = offset + thid - prefixSum[thid] - prefixSums[blockId];
        }
        destination[place] = source[thid];
    }

}


template<int Threads, int Elements>
__device__ __forceinline__
void pref_sum_dev(int *data, int *offsetPrefSums, int size) {
    int const BlockSize = Threads * Elements;
    __shared__ int shared[2][BlockSize];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * BlockSize;

    data += offset;
    size -= offset;

    for (int thid = threadIdx.x; thid < BlockSize; thid += Threads) {
        shared[0][thid] = thid < size ? data[thid] : 0;
    }
    __syncthreads();

    pref_sum_and_move_result_dev<Threads, Elements>(shared, data, offsetPrefSums, size, blockId);
}


extern "C" {
__global__
void count_and_pref_sum(int *in, int *out, int *prefixSums, int size, int mask) {
    count_and_pref_sum_dev<RADIX_THREADS, RADIX_ELEMENTS>(in, out, prefixSums, size, mask);
}

__global__
void sort(int *source, int *destination, int *prefixSum, int *prefixSums, int mask, int n, int blocks_num) {
    sort_dev<RADIX_THREADS, RADIX_ELEMENTS>(source, destination, prefixSum, prefixSums, mask, n,
                                            n - prefixSums[blocks_num]);

}


__global__
void one_block_prefsum(int *data, int size) {
    __shared__ int shared[2][RADIX_BLOCK];

    for (int thid = threadIdx.x; thid < RADIX_BLOCK; thid += RADIX_THREADS) {
        shared[0][thid] = thid < size ? data[thid] : 0;
    }

    __syncthreads();
    bool to = 0;
    prefixSumDev<RADIX_THREADS, RADIX_ELEMENTS>(shared, to);

    for (int thid = threadIdx.x; thid < RADIX_BLOCK && thid < size; thid += RADIX_THREADS) {
        data[thid] = shared[to][thid];
    }
}

__global__
void pref_sum(int *data, int *offsetPrefSums, int size) {
    pref_sum_dev<RADIX_THREADS, RADIX_ELEMENTS>(data, offsetPrefSums, size);
}
__global__
void add(int *data, int *offsetPrefSums, int size) {
    int blockId = one_dimension_blockId();
    size -= blockId * RADIX_BLOCK;
    data += blockId * RADIX_BLOCK;
    for (int thid = threadIdx.x; thid < RADIX_BLOCK && thid < size; thid += RADIX_THREADS) {
        data[thid] += offsetPrefSums[blockId];
    }
}

}



