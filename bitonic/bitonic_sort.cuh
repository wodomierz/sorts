#ifndef SORTS_BITONIC_SORT_CUH
#define SORTS_BITONIC_SORT_CUH

#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"

template<int ThreadsPow>
__device__ __forceinline__
void bitonic_merge_device(int *to_sort, int size, int tab[]) {
    const int THREADS = (1 << ThreadsPow);
    const int LOCAL_BLOCK = (THREADS * 2);

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

        min_max(tab, wireThid, opposite, LOCAL_BLOCK);
        __syncthreads();
        for (int d_power = d_half_traingle_p - 1; d_power >= 0; d_power--) {
            int wireThid = threadId + ((threadId >> d_power) << d_power);
            int opposite = wireThid + (1 << d_power);
            min_max(tab, wireThid, opposite, LOCAL_BLOCK);
            __syncthreads();
        }
    }
    if (threadId < size) to_sort[threadId] = tab[threadId];
    if (threadId + THREADS < size) to_sort[threadId + THREADS] = tab[threadId + THREADS];
}

#endif //SORTS_BITONIC_SORT_CUH