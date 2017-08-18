
#include "cuda_device.h"

#ifndef SORTS_KERNEL_COMMONS_CUH
#define SORTS_KERNEL_COMMONS_CUH


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
int gerOrInf(int *to_sort, int index, int size) {
    return index < size ? to_sort[index] : 2147483647; //max int
}

template <typename T>
__device__ __forceinline__
void swap(T& a, T&b) {
    T c  = a;
    a = b;
    b = c;
}

#endif
