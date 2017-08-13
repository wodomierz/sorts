//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_QUICK_SORT_H
#define SORTS_QUICK_SORT_H

#include <cuda.h>
#include "../utils/utils.h"

#define QUICKTHREADS_POW 8
#define QUICK_THREADS_IN_BLOCK 256
#define QUICK_BLOCK_SIZE QUICK_THREADS_IN_BLOCK
#define OTHER_SORT_LIM 512
#define MAX_SEQ 1024

void quick_sort(int* to_sort, int size);

#endif //SORTS_QUICK_SORT_H
