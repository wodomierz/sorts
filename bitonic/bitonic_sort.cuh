#ifndef SORTS_BITONIC_SORT_CUH
#define SORTS_BITONIC_SORT_CUH

#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"




__device__ __forceinline__
void bitonic_merge_device_phase2(int size, int tab[], int d_power) {
    for (; d_power >= 0; d_power--) {
        int wireThid = threadIdx.x + ((threadIdx.x >> d_power) << d_power);
        int opposite = wireThid + (1 << d_power);
        min_max(tab, wireThid, opposite, size);
        __syncthreads();
    }
}

__device__ __forceinline__
void bitonic_merge_device_phase2(int tab[], int d_power) {
    for (; d_power >= 0; d_power--) {
        int wireThid = threadIdx.x + ((threadIdx.x >> d_power) << d_power);
        int opposite = wireThid + (1 << d_power);
        min_max(tab, wireThid, opposite);
        __syncthreads();
    }
}


template<int ThreadsPow>
__device__ __forceinline__
void bitonic_merge_device_phase2_global(int* to_sort) {
    const int Threads = (1 << ThreadsPow);
    __shared__ int tab[Threads*2];
    int threadId = threadIdx.x;

    tab[threadId] = to_sort[threadId];
    tab[threadId + Threads] = to_sort[threadId + Threads];

    __syncthreads();
    bitonic_merge_device_phase2( tab, ThreadsPow);

    to_sort[threadId] = tab[threadId];
    to_sort[threadId + Threads] = tab[threadId + Threads];
}



template<int ThreadsPow>
__device__ __forceinline__
void bitonic_merge_device(int *to_sort, int size, int tab[]) {
    const int THREADS = (1 << ThreadsPow);
    int threadId = threadIdx.x;

    tab[threadId] = gerOrInf(to_sort, threadId, size);
    tab[threadId + THREADS] = gerOrInf(to_sort, threadId + THREADS, size);

    __syncthreads();

    for (int d_triangle = 2, d_half_traingle_p = 0;
         d_half_traingle_p <= ThreadsPow;
         d_half_traingle_p++, d_triangle <<= 1) {

        int wireThid = threadId + ((threadId >> d_half_traingle_p) << d_half_traingle_p);

        int local_thid = wireThid & (d_triangle - 1);
        int opposite = wireThid - local_thid + d_triangle - 1 - local_thid;

        min_max(tab, wireThid, opposite, size);
        __syncthreads();
        bitonic_merge_device_phase2(size, tab,d_half_traingle_p - 1);
    }
    if (threadId < size) to_sort[threadId] = tab[threadId];
    if (threadId + THREADS < size) to_sort[threadId + THREADS] = tab[threadId + THREADS];
}


template<int ThreadsPow>
__device__ __forceinline__
void bitonic_merge_device(int *to_sort, int tab[]) {
    const int Threads = (1 << ThreadsPow);
    int threadId = threadIdx.x;

    tab[threadId] = to_sort[threadId];
    tab[threadId + Threads] = to_sort[threadId + Threads];

    __syncthreads();

    for (int d_triangle = 2, d_half_traingle_p = 0;
         d_half_traingle_p <= ThreadsPow;
         d_half_traingle_p++, d_triangle <<= 1) {

        int wireThid = threadId + ((threadId >> d_half_traingle_p) << d_half_traingle_p);

        int local_thid = wireThid & (d_triangle - 1);
        int opposite = wireThid - local_thid + d_triangle - 1 - local_thid;

        min_max(tab, wireThid, opposite);
        __syncthreads();
        bitonic_merge_device_phase2(tab,d_half_traingle_p - 1);
    }
    to_sort[threadId] = tab[threadId];
    to_sort[threadId + Threads] = tab[threadId + Threads];
}


#endif //SORTS_BITONIC_SORT_CUH