#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"
#include "bitonic_sort.cuh"
#include "bitonic_sort.h"

extern "C" {

__global__
void bitonic_merge2(int *to_sort) {
    __shared__ int tab[BITONIC_BLOCK];
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * BITONIC_BLOCK;
    bitonic_merge_device<BITONIC_THREADS_POW>(to_sort + offset, tab);
}

__global__
void bitonic_merge(int *to_sort, int d_power, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;

    int wireThid = thid + ((thid >> d_power) << d_power);

    int opposite = wireThid + (1 << d_power);
    min_max(to_sort, wireThid, opposite, size);
}

__global__
void bitonic_triangle_merge(int *to_sort, int half_triangle_power, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;

    int d_triangle = 1 << (half_triangle_power + 1);
    int wireThid = thid + ((thid >> half_triangle_power) << half_triangle_power);
    int local_thid = wireThid & (d_triangle - 1);
    int opposite = wireThid - local_thid + d_triangle - 1 - local_thid;
    min_max(to_sort, wireThid, opposite, size);
}

__global__
void phase2_global(int *to_sort) {
    int blockId = one_dimension_blockId();
    int offset = blockId * BITONIC_BLOCK;
    bitonic_merge_device_phase2_global<BITONIC_THREADS_POW>(to_sort + offset);
}

}


