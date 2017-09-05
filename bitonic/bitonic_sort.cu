#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"
#include "bitonic_sort.cuh"


#define BIT_THREADS_POW 10
#define BIT_THREADS (1 << BIT_THREADS_POW)
#define BIT_BLOCK (2* BIT_THREADS)

extern "C" {



__global__
void bitonic_merge2(int *to_sort) {
    __shared__ int tab[BIT_BLOCK];
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * blockDim.x * 2;
    bitonic_merge_device<BIT_THREADS_POW>(to_sort + offset, tab);
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
    int offset = blockId * blockDim.x * 2;
    bitonic_merge_device_phase2_global<BIT_THREADS_POW>(to_sort + offset);
}

}


