
#ifndef SORTS_QUICK_SORT_H
#define SORTS_QUICK_SORT_H

#include <cuda.h>
#include "../utils/utils.h"

#define QUICK_THREADS_POW 10
#define QUICK_THREADS (1<< QUICK_THREADS_POW)
#define QUICK_ELEMENTS_PER_THREAD 16
#define QUICK_BLOCK_SIZE (QUICK_THREADS * QUICK_ELEMENTS_PER_THREAD)

void quick_sort(int *to_sort, int size);

#endif //SORTS_QUICK_SORT_H
