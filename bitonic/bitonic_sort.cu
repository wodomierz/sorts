//#include <cstdio>
#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"
#include "bitonic_sort.cuh"

extern "C" {

__global__
void bitonic_merge2(int *to_sort) {
    __shared__ int tab[2048];
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int offset = blockId * blockDim.x*2;
    bitonic_merge_device<10>(to_sort + offset,2048, tab);
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
    int local_thid = wireThid & ( d_triangle-1);
    int opposite = wireThid - local_thid + d_triangle - 1 - local_thid;
    min_max(to_sort, wireThid, opposite, size);
}

}


