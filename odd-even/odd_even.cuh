#ifndef SORTS_ODD_EVEN_CUH
#define SORTS_ODD_EVEN_CUH

#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"

template<int ThreadsPow>
__device__ __forceinline__
void odd_even_device(int *to_sort, int tab[]) {
    //I assume, that to_sort is aligned to blocksize
    const int Threads = (1 << ThreadsPow);

    int thid = threadIdx.x;
    tab[thid] = to_sort[thid];
    tab[thid + Threads] = to_sort[thid + Threads];
    __syncthreads();

    for (int pow__half_batch = 0, half_batch = 1;
         pow__half_batch <= ThreadsPow;
         pow__half_batch++, half_batch <<= 1) {

        int wireThid = thid + ((thid >> pow__half_batch) << pow__half_batch);
        int opposite = wireThid + half_batch;
        //is size check needed here?
        min_max(tab, wireThid, opposite, Threads * 2);
        __syncthreads();
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {

            int d = 1 << d_power;

            int period = half_batch - d;

            int wire_id = thid + (((thid >> d_power) + ((thid / period) << 1) + 1) << d_power);
            int opposite = wire_id + d;
            min_max(tab, wire_id, opposite, Threads * 2);

            __syncthreads();
        }

    }

    to_sort[thid] = tab[thid];
    to_sort[thid + Threads] = tab[thid + Threads];

}

#endif //SORTS_ODD_EVEN_CUH