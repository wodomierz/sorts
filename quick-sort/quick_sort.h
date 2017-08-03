//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_QUICK_SORT_H
#define SORTS_QUICK_SORT_H

#include <cuda.h>

#define QUICK_THREADS_IN_BLOCK 1024
#define QUICK_BLOCK_SIZE QUICK_THREADS_IN_BLOCK
#define OTHER_SORT_LIM 2048
#define MAX_SEQ 1024 //????

struct DevArray {
    //inclusive
    const int start;
    //exclusive
    const int end;
    const int size = end - start; //TODO check
    const CUdeviceptr d;

    DevArray(CUdeviceptr d, int start, int end) : d(d), start(start), end(end) {}
};

struct WorkUnit {
    DevArray d;
    int pivot;

    WorkUnit(DevArray d, int pivot) : d(d), pivot(pivot) {}

};

#endif //SORTS_QUICK_SORT_H
