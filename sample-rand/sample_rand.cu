#include "sample_rand.h"
#include "../prefsum/prefsum.cuh"


#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"
#include "sample_rand.cuh"


extern "C" {


__device__ __forceinline__
int place(int size, int seed, int plus, int i) {
    return (i * seed + plus) % size;
}


__global__
void chujowy_sort(int *to_sort, int size) {
    chujowy_sort_dev(to_sort, size);
}


__global__
void sample(int *tab, int size, int seed, int plus, int *bst) {
    __shared__ int to_sort[SAMPLE_BLOCK];
    int thid = threadIdx.x;

    int i = thid;
    for (;i < SAMPLE_BLOCK; i+= SAMPLE_THREADS) {
        to_sort[i] = tab[place(size, seed, plus, i)];
    }

    __syncthreads();

    chujowy_sort_dev(to_sort, SAMPLE_BLOCK);
    __syncthreads();
    if (threadIdx.x ==0) {
        int iteratr = 0;
        for (int i = 2; i <= S_SIZE; i *= 2) {
            for (int j = 1; j < i; j += 2) {
                bst[iteratr++] = to_sort[(j * (S_SIZE / i) - 1)*AS];
            }
        }
        bst[S_SIZE - 1] = 0;
    }
}


__device__ __forceinline__
int findIndex(int e, int *bst) {
    int j = 1;
    int k = S_POW;
    while (k--) {
        j = 2 * j + (e > bst[j - 1]);
    }
    j = j - S_SIZE; // bucket index
    return j;
}


__global__
void prefsum1(int *localPrefsums, int *maxPrefSums, int number_of_counter_blocks, int *sample_offsets, int size) {
    prefsum1_dev<PREFSUM_THREADS, 2>(localPrefsums, maxPrefSums, number_of_counter_blocks, sample_offsets, size);
}

__global__
void prefsum(int *localPrefsums, int *maxPrefSums, int size) {
    prefsum_dev<PREFSUM_THREADS, 2>(localPrefsums, maxPrefSums, size);
}

__global__
void counters(int *to_sort, int *sample, int *prefsums, int number_of_blocks, int size) {
    __shared__ int bst[S_SIZE];
    __shared__ int histogram[A][S_SIZE];


    int x = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int gthid = x + y * gridDim.x * blockDim.x * ELEMENTS_PER_THREAD;

    int threadId = threadIdx.x;

    if (threadId < S_SIZE) { //?
        bst[threadId] = sample[threadId];
        for (int a = 0; a < A; ++a) {
            histogram[a][threadId] = 0;
        }
    }
    __syncthreads();


    for (int i = 0; i < ELEMENTS_PER_THREAD && gthid + i * THREADS_PER_BLOCK < size; ++i) {
//    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        //ke?

        int j = findIndex(to_sort[gthid + i * THREADS_PER_BLOCK], bst);
        atomicAdd(histogram[gthid % A] + j, 1);
    }
    __syncthreads();


    if (threadId < S_SIZE) {
        //bug?
        for (int i = 1; i < A; ++i) {
            histogram[0][threadId] += histogram[i][threadId];
        }
        int index = (threadId * number_of_blocks) + blockId;
        atomicExch(prefsums + index, histogram[0][threadId]);
    }
}

__global__
void scatter(int *in, int *out, int *sample, int *prefsums, int number_of_blocks, int size) {
    __shared__ int bst[S_SIZE];
    __shared__ int histogram[S_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int gthid = x + y * gridDim.x * BLOCK_SIZE;
    int threadId = threadIdx.x;
    if (threadId < S_SIZE) { //?
        bst[threadId] = sample[threadId];
        histogram[threadId] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ELEMENTS_PER_THREAD && gthid + i * THREADS_PER_BLOCK < size; ++i) {
        //ke?
        int e = in[gthid + i * THREADS_PER_BLOCK];
        int j = findIndex(e, bst);
        int local_index = atomicAdd(histogram + j, 1);
        int indexInPrefsums = (j * number_of_blocks) + blockId;
        int offset = 0;
        if (indexInPrefsums > 0) {
            offset = prefsums[indexInPrefsums - 1];
        }
        out[offset + local_index] = e;
    }
}

}