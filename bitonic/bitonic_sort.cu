#include <cstdio>


extern "C" {

__device__
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
} ;

__global__
void bitonic_merge(int *to_sort, int d, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;
    int wireThid = thid + (d) * (thid / d);

    int opposite = wireThid + d;
    min_max(to_sort, wireThid, opposite, size);
}

__global__
void bitonic_triangle_merge(int *to_sort, int d_traingle, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;

    int halftriangle = d_traingle / 2;
    int wireThid = thid + halftriangle * (thid / halftriangle);

    int local_thid = wireThid % d_traingle;
    int opposite = wireThid - local_thid + d_traingle - 1 - local_thid;
    min_max(to_sort, wireThid, opposite, size);
}

}


