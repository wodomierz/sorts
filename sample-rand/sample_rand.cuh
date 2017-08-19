#ifndef SORTS_SAMPLE_RAND_CUH
#define SORTS_SAMPLE_RAND_CUH

#include "../utils/kernel_commons.cuh"
#include "../utils/cuda_device.h"
#include "../prefsum/prefsum.cuh"


template <int SamplePow>
__device__ __forceinline__
int findIndex(int e, int *bst) {
    const int SampleSize = (1 << SamplePow);
    int j = 1;
    int k = SamplePow;
    while (k--) {
        j = 2 * j + (e > bst[j - 1]);
    }
    j = j - SampleSize; // bucket index
    return j;
}


template <int THREADS, int ELEMENTS>
__device__ __forceinline__
void offset_count(int *localPrefsums, int *sample_offsets, int number_of_counter_blocks, int size) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int thid = threadIdx.x + blockId * THREADS * ELEMENTS;

    for (int i = 0; i < ELEMENTS && thid < size; ++i, thid += THREADS) {
        if (thid % number_of_counter_blocks == number_of_counter_blocks - 1) {
            sample_offsets[thid / number_of_counter_blocks + 1] = localPrefsums[thid];
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

template <int Threads, int Elements, int SamplePow, int ArraysNum>
__device__ __forceinline__
void counters_dev(int *to_sort, int *sample, int *prefsums, int number_of_blocks, int size) {
    const int SampleSize = 1 << SamplePow;
    __shared__ int bst[SampleSize];
    __shared__ int histogram[ArraysNum][SampleSize];


    int x = blockIdx.x * blockDim.x * Elements + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int gthid = x + y * gridDim.x * blockDim.x * Elements;

    int threadId = threadIdx.x;

    if (threadId < SampleSize) { //?
        bst[threadId] = sample[threadId];
        for (int a = 0; a < ArraysNum; ++a) {
            histogram[a][threadId] = 0;
        }
    }
    __syncthreads();

    for (int i = 0; i < Elements && gthid + i * Threads < size; ++i) {
        int j = findIndex<SamplePow>(to_sort[gthid + i * Threads], bst);
        atomicAdd(histogram[gthid % ArraysNum] + j, 1);
    }
    __syncthreads();


    if (threadId < SampleSize) {
        //bug?
        for (int i = 1; i < ArraysNum; ++i) {
            histogram[0][threadId] += histogram[i][threadId];
        }
        int index = (threadId * number_of_blocks) + blockId;
        atomicExch(prefsums + index, histogram[0][threadId]);
    }
}

//SampleSize should be less than Threads!!!
template <int Threads, int Elements, int SamplePow >
__device__ __forceinline__
void scatter_dev(int *in, int *out, int *sample, int *prefsums, int number_of_blocks, int size) {
    const int SampleSize = 1 << SamplePow;
    const int BlockSize = Elements * Threads;

    __shared__ int bst[SampleSize];
    __shared__ int histogram[SampleSize];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x;
    int thid_offset = blockId *BlockSize;
    in += thid_offset;
    size -= thid_offset;

    if (threadId < SampleSize) { //?
        bst[threadId] = sample[threadId];
        histogram[threadId] = 0;
    }
    __syncthreads();

    for (; threadId < BlockSize && threadId < size; threadId += Threads) {
        int e = in[threadId];
        int j = findIndex<SamplePow>(e, bst);
        int local_index = atomicAdd(histogram + j, 1);
        int indexInPrefsums = (j * number_of_blocks) + blockId;
        int offset = 0;
        if (indexInPrefsums > 0) {
            offset = prefsums[indexInPrefsums - 1];
        }
        out[offset + local_index] = e;
    }
}

#endif //SORTS_SAMPLE_RAND_CUH