//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_QUICK_SORT_DEVICE_H
#define SORTS_QUICK_SORT_DEVICE_H


#include <cuda.h>

class quick_sort_device {

    void quick_phase1(CUdeviceptr toSort);
    quick_sort_device();
};


#endif //SORTS_QUICK_SORT_DEVICE_H
