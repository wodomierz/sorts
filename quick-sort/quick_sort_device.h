//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_QUICK_SORT_DEVICE_H
#define SORTS_QUICK_SORT_DEVICE_H


#include <cuda.h>
#include <altivec.h>
#include <vector>
#include "quick_sort.h"

namespace quick {
    class Device {
        CUmodule cuModule;
        CUfunction gqsortDev;
        CUfunction lqsortDev;
    public:
        Device();
        void gqsort(Block* blocks,int blocks_count, CUdeviceptr in, CUdeviceptr out, WorkUnit* news);
        void lqsort(DevArray* seqs, int seq_count, CUdeviceptr& to_sort, CUdeviceptr& out);
    };
}


#endif //SORTS_QUICK_SORT_DEVICE_H
