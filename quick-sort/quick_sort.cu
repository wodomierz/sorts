
#include "quick_sort.h"
#include "quick_sort.cuh"
#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"
#include "quick_shared.h"
#include "../bitonic/bitonic_sort.cuh"
#include "../prefsum/prefsum.cuh"

extern "C" {

__global__
void pivot(int* array, int size, int* result) {
    *result = median(array, 0, size);
}

__global__
void gqsort(Block *blocks, int *in, int *out, WorkUnit *news) {
    gqsort_dev<QUICKTHREADS_POW>(blocks, in, out, news);
}


__global__
void lqsort(DevArray *seqs, int *in_h, int *out_h) {
    lqsort_dev<QUICKTHREADS_POW>(seqs, in_h, out_h);
}

}