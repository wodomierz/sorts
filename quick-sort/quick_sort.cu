
#include "quick_sort.h"
#include "quick_sort.cuh"

extern "C" {

__global__
void pivot(int *array, int size, int *result) {
    *result = median(array, 0, size);
}

__global__
void gqsort(Block *blocks, int *in, int *out, WorkUnit *news) {
    gqsort_dev<QUICK_THREADS_POW>(blocks, in, out, news);
}


__global__
void lqsort(DevArray *seqs, int *in_h, int *out_h) {
    lqsort_dev<QUICK_THREADS_POW>(seqs, in_h, out_h);
}

}