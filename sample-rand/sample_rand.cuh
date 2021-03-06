#ifndef SORTS_SAMPLE_RAND_CUH
#define SORTS_SAMPLE_RAND_CUH

#include "../utils/kernel_commons.cuh"
#include "../utils/cuda_device.h"
#include "../prefsum/prefsum.cuh"


template<int SamplePow>
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


template<int THREADS, int ELEMENTS>
__device__ __forceinline__
void offset_count(int *localPrefsums, int *sample_offsets, int number_of_counter_blocks, int size) {
    int blockId = one_dimension_blockId();

    int thid = threadIdx.x + blockId * THREADS * ELEMENTS;

    for (int i = 0; i < ELEMENTS && thid < size; ++i, thid += THREADS) {
        if (thid % number_of_counter_blocks == number_of_counter_blocks - 1) {
            sample_offsets[thid / number_of_counter_blocks + 1] = localPrefsums[thid];
        }
    }
}

template<int Threads, int Elements>
__device__ __forceinline__
void prefsum1_dev(int *localPrefsums, int *maxPrefSums, int number_of_counter_blocks, int *sample_offsets, int size) {
    int blockId = one_dimension_blockId();
    int thid = threadIdx.x + blockId * Threads * Elements;

    int global_offset = maxPrefSums[blockId];

    for (int i = 0; i < Elements && thid < size; ++i, thid += Threads) {
        localPrefsums[thid] += global_offset;
    }
    offset_count<Threads, Elements>(localPrefsums, sample_offsets, number_of_counter_blocks, size);
}


template<int Threads, int Elements>
__device__ __forceinline__
void prefsum_dev(int *localPrefsums, int *maxPrefSums, int size) {

    const int BlockSize = (Threads * Elements);

    __shared__ int shared[2][BlockSize];

    int blockId = one_dimension_blockId();
    int offset = blockId * BlockSize;

    localPrefsums += offset;
    size -= offset;

    for (int thid = threadIdx.x; thid < BlockSize; thid += Threads) {
        shared[0][thid] = get_or_zero(localPrefsums, thid, size);
    }
    __syncthreads();
    pref_sum_and_move_result_dev<Threads, Elements>(shared, localPrefsums, maxPrefSums, size, blockId);
}

template<int Threads, int Elements, int SamplePow, int ArraysNum>
__device__ __forceinline__
void counters_dev(int *to_sort, int *sample, int *prefsums, int number_of_blocks, int size) {
    const int SampleSize = 1 << SamplePow;
    const int BlockSize = Threads * Elements;
    __shared__ int bst[SampleSize];
    __shared__ int histogram[ArraysNum][SampleSize];

    int blockId = one_dimension_blockId();
    int offset = blockId * BlockSize;
    to_sort += offset;
    size -= offset;

    int threadId = threadIdx.x;

    if (threadId < SampleSize) { //?
        bst[threadId] = sample[threadId];
        for (int a = 0; a < ArraysNum; ++a) {
            histogram[a][threadId] = 0;
        }
    }
    __syncthreads();

    for (int thid = threadId; thid < BlockSize && thid < size; thid += Threads) {
        int j = findIndex<SamplePow>(to_sort[thid], bst);
        atomicAdd(histogram[thid % ArraysNum] + j, 1);
    }
    __syncthreads();


    if (threadId < SampleSize) {
        for (int i = 1; i < ArraysNum; ++i) {
            histogram[0][threadId] += histogram[i][threadId];
        }
        int index = (threadId * number_of_blocks) + blockId;
        atomicExch(prefsums + index, histogram[0][threadId]);
    }
}

//SampleSize should be less than Threads!!!
template<int Threads, int Elements, int SamplePow>
__device__ __forceinline__
void scatter_dev(int *in, int *out, int *sample, int *prefsums, int number_of_blocks, int size) {
    const int SampleSize = 1 << SamplePow;
    const int BlockSize = Elements * Threads;

    __shared__ int bst[SampleSize];
    __shared__ int histogram[SampleSize];

    int blockId = one_dimension_blockId();
    int threadId = threadIdx.x;
    int thid_offset = blockId * BlockSize;
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


__device__ __forceinline__
int place(int size, int seed, int plus, int i) {
    return (i * seed + plus) % size;
}

template<int BlockSize, int Threads, int SamplSize, int A>
__device__ __forceinline__
void sample_dev(int *tab, int size, int seed, int plus, int *bst) {
    __shared__ int to_sort[BlockSize];
    int thid = threadIdx.x;

    int i = thid;
    for (; i < BlockSize; i += Threads) {
        to_sort[i] = tab[place(size, seed, plus, i)];
    }

    __syncthreads();

    dummy_sort_dev(to_sort, BlockSize);
    __syncthreads();
    if (threadIdx.x == 0) {
        int iteratr = 0;
        for (int i = 2; i <= SamplSize; i *= 2) {
            for (int j = 1; j < i; j += 2) {
                bst[iteratr++] = to_sort[(j * (SamplSize / i) - 1) * A];
            }
        }
        bst[SamplSize - 1] = 0;
    }
}



#endif //SORTS_SAMPLE_RAND_CUH