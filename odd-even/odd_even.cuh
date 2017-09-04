#ifndef SORTS_ODD_EVEN_CUH
#define SORTS_ODD_EVEN_CUH

#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"

template<int THREADS_POW>
__device__ __forceinline__
void odd_even_device(int *to_sort, int size, int tab[]) {
    const int THREADS = (1 << THREADS_POW);

    int thid = threadIdx.x;
    tab[thid] = gerOrInf(to_sort, thid, size);
    tab[thid + THREADS] = gerOrInf(to_sort, thid + THREADS, size);
    __syncthreads();

    for (int pow__half_batch = 0, half_batch = 1;
         pow__half_batch <= THREADS_POW;
         pow__half_batch++, half_batch <<= 1) {

        int wireThid = thid + ((thid >> pow__half_batch) << pow__half_batch);
        int opposite = wireThid + half_batch;
        //is size check needed here?
        min_max(tab, wireThid, opposite, size);
        __syncthreads();
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {

            int d = 1 << d_power;

            int period = half_batch - d;

            int wire_id = thid + (((thid >> d_power) + ((thid / period) << 1) + 1) << d_power);
            int opposite = wire_id + d;
            min_max(tab, wire_id, opposite, size);

            __syncthreads();
        }

    }

    if (thid < size) to_sort[thid] = tab[thid];
    if (thid + THREADS < size) to_sort[thid + THREADS] = tab[thid + THREADS];

}

#endif //SORTS_ODD_EVEN_CUH