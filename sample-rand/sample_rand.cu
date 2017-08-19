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
    counters_dev<THREADS_PER_BLOCK, ELEMENTS_PER_THREAD, S_POW, ARRAYS_NUM>(to_sort, sample, prefsums, number_of_blocks, size);
}

__global__
void scatter(int *in, int *out, int *sample, int *prefsums, int number_of_blocks, int size) {
    scatter_dev<THREADS_PER_BLOCK,ELEMENTS_PER_THREAD, S_POW>(in, out, sample, prefsums, number_of_blocks, size);
}

}