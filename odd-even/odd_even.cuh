#include "../utils/cuda_device.h"

__device__
template <int THREADS>
void odd_even(int *to_sort) {
    __shared__ int tab[2*THREADS];

    int x = blockIdx.x * blockDim.x *2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = threadIdx.x;
    int gthid = x + y * gridDim.x * blockDim.x * 2;
    //TODO check *2 here

    tab[thid] = to_sort[gthid];
    tab[thid + THREADS] = to_sort[gthid + THREADS];
    __syncthreads();

    for (int pow__half_batch = 0, half_batch = 1;
         pow__half_batch <= 10;
         pow__half_batch++, half_batch <<=1) {

        int wireThid = thid + ((thid >> pow__half_batch) << pow__half_batch);
        int opposite = wireThid + half_batch;
        min_max(tab, wireThid, opposite, 2*THREADS);
        __syncthreads();
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {

            int d = 1 << d_power;

            int period = half_batch - d;

            int wire_id = thid + (((thid>>d_power) + ((thid / period) << 1) + 1) << d_power);
            int opposite = wire_id + d;
            min_max(tab, wire_id, opposite, 2*THREADS);

            __syncthreads();
        }

    }

    to_sort[gthid] = tab[thid];
    to_sort[gthid + THREADS] = tab[thid + THREADS];

}