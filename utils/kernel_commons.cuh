
#include "cuda_device.h"

#ifndef SORTS_KERNEL_COMMONS_CUH
#define SORTS_KERNEL_COMMONS_CUH


__device__ __forceinline__
int one_dimension_blockId() {
    return blockIdx.x + blockIdx.y * gridDim.x;
}

template<int BlockSize>
__device__ __forceinline__
int one_dimension_offset() {
    return one_dimension_blockId() * BlockSize;
}

__device__ __forceinline__
void min_max(int *tab, int for_min, int for_max, int size) {
    if (for_min >= size || for_max >= size) {
        return;
    }
    int min = tab[for_min];
    int max = tab[for_max];
    if (max < min) {
        atomicExch(tab + for_max, min);
        atomicExch(tab + for_min, max);
    }
};

__device__ __forceinline__
void min_max(int *tab, int for_min, int for_max) {
    int min = tab[for_min];
    int max = tab[for_max];
    if (max < min) {
        atomicExch(tab + for_max, min);
        atomicExch(tab + for_min, max);
    }
};



__device__ __forceinline__
void dummy_sort_dev(int *to_sort, int size) {
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < size; ++i) {
            for (int j = 0; j < i; ++j) {
                min_max(to_sort, j, i, size);
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__
int get_or_inf(int *to_sort, int index, int size) {
    return index < size ? to_sort[index] : 2147483647; //max int
}

__device__ __forceinline__
int get_or_zero(int *tab, int i, int size) {
    return i < size ? tab[i] : 0;
}


template<typename T>
__device__ __forceinline__
void swap(T &a, T &b) {
    T c = a;
    a = b;
    b = c;
}

#endif
