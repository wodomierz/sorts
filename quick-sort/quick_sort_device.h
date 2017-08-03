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
    public:
        void quick_phase1(CUdeviceptr toSort);

        Device();

        std::vector<WorkUnit> gqsort();
    };
}


#endif //SORTS_QUICK_SORT_DEVICE_H
