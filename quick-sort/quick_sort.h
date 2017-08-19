//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_QUICK_SORT_H
#define SORTS_QUICK_SORT_H

#include <cuda.h>
#include "../utils/utils.h"

#define QUICKTHREADS_POW 9

void quick_sort(int* to_sort, int size);

#endif //SORTS_QUICK_SORT_H
