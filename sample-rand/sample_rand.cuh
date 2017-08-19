#ifndef SORTS_SAMPLE_RAND_CUH
#define SORTS_SAMPLE_RAND_CUH

#include "../utils/kernel_commons.cuh"
#include "../utils/cuda_device.h"
#include "../prefsum/prefsum.cuh"

template <int THREADS, int ELEMENTS>
__device__ __forceinline__
void offset_count(int *localPrefsums, int *sample_offsets, int number_of_counter_blocks, int size) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int thid = threadIdx.x + blockId * THREADS * ELEMENTS;

    for (int i = 0; i < ELEMENTS && thid < size; ++i, thid += THREADS) {
        if (thid % number_of_counter_blocks == number_of_counter_blocks - 1) {
            sample_offsets[thid / number_of_counter_blocks + 1]
                = localPrefsums[thid];
        }
    }
}

template <int THREADS, int ELEMENTS>
__device__ __forceinline__
void prefsum1_dev(int *localPrefsums, int *maxPrefSums, int number_of_counter_blocks, int *sample_offsets, int size) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int thid = threadIdx.x + blockId * THREADS * ELEMENTS;

    int global_offset = maxPrefSums[blockId];

    for (int i = 0; i < ELEMENTS && thid < size; ++i, thid += THREADS) {
        localPrefsums[thid] += global_offset;
    }
    offset_count<THREADS, ELEMENTS>(localPrefsums, sample_offsets, number_of_counter_blocks, size);
}




template <int THREADS, int ELEMENTS>
__device__ __forceinline__
void prefsum_dev(int *localPrefsums, int *maxPrefSums, int size) {

    const int LOCAL_BLOCK_SIZE = (THREADS * ELEMENTS);

    __shared__ int shared[2][LOCAL_BLOCK_SIZE];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * LOCAL_BLOCK_SIZE;

    localPrefsums += offset;
    size -= offset;

    for (int thid= threadIdx.x; thid < LOCAL_BLOCK_SIZE; thid += THREADS) {
        shared[0][thid] = getOrZero(localPrefsums, thid, size);
    }
    __syncthreads();

    bool to = 0;
    prefixSumDev<THREADS, 2>(shared, to);

    for (int thid= threadIdx.x; thid < LOCAL_BLOCK_SIZE && thid < size; thid += THREADS) {
        localPrefsums[thid] = shared[to][thid];
    }
    if (ELEMENTS * threadIdx.x + 1 == LOCAL_BLOCK_SIZE -1) {
        maxPrefSums[blockId + 1] = shared[to][2 * threadIdx.x + 1];
    }

}


#endif //SORTS_SAMPLE_RAND_CUH