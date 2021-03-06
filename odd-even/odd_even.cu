#include "../utils/kernel_commons.cuh"
#include "odd_even.cuh"
#include "odd_even.h"

extern "C" {

__global__
void odd_even(int *to_sort) {
    __shared__ int tab[ODD_EVEN_BLOCK];
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * ODD_EVEN_BLOCK;
    odd_even_device<ODD_EVEN_THREADS_POW>(to_sort + offset, tab);
}

__global__
void odd_even_phase1(int *to_sort, int d_pow, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;

    int wireThid = thid + ((thid >> d_pow) << d_pow);

    int opposite = wireThid + (1 << d_pow);
    min_max(to_sort, wireThid, opposite, size);
}


__global__
void odd_even_phase2(int *to_sort, int d_power, int half_batch_size, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;

    int d = 1 << d_power;

    int period = half_batch_size - d;

    int wire_id = thid + (((thid >> d_power) + ((thid / period) << 1) + 1) << d_power);
    int opposite = wire_id + d;
    min_max(to_sort, wire_id, opposite, size);

}

}




