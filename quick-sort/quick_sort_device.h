#ifndef SORTS_QUICK_SORT_DEVICE_H
#define SORTS_QUICK_SORT_DEVICE_H


#include "quick_sort.h"
#include "quick_shared.h"
#include <cuda.h>

namespace quick {
    class Device {
        CUmodule cuModule;
        CUfunction gqsortDev;
        CUfunction lqsortDev;
        CUfunction pivotDev;
    public:
        Device();

        int pivot(CUdeviceptr to_sort, int size);

        void gqsort(Block *blocks, int blocks_count, CUdeviceptr in, CUdeviceptr out, WorkUnit *news);

        void lqsort(DevArray *seqs, int seq_count, CUdeviceptr &to_sort, CUdeviceptr &buffer);
    };
}


#endif //SORTS_QUICK_SORT_DEVICE_H
