//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_QUICK_SORT_DEVICE_H
#define SORTS_QUICK_SORT_DEVICE_H


#include <cuda.h>

namespace quick {
    class Device {

        void quick_phase1(CUdeviceptr toSort);

        Device();
    };
}


#endif //SORTS_QUICK_SORT_DEVICE_H
