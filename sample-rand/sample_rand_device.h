//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_SAMPLE_RAND_DEVICE_H
#define SORTS_SAMPLE_RAND_DEVICE_H


#include <cuda.h>
#include "sample_rand_context.h"
#include "PrefsumContext.h"

namespace sample_rand {
    class Device {
    public:
        CUfunction cuOdeven;

        CUdevice cuDevice;
        CUmodule cuModule;
        CUcontext cuContext;
        CUfunction prefsumDev;
        CUfunction sumSums;
        CUfunction prefsumDev1;
        CUfunction countersCU;
        CUfunction scatterCU;

        CUfunction dummy_sortDev;


        CUfunction sampleDev;

        Device();

        void localPrefSums(sample_rand::Context &context, sample_rand::PrefsumContext &prefsumMemory, CUstream cUstream);

        void prefsumOfBatchSums(sample_rand::PrefsumContext &prefsumMemory, CUstream cUstream);

        void globalPrefSums(sample_rand::Context &memory, sample_rand::PrefsumContext &prefsumMemory, CUstream cUstream);

        void chujowy(sample_rand::Context &memory);

        void scatter(sample_rand::Context &memory, CUstream);

        void counters(sample_rand::Context &memory, CUstream);

        void odd_even(sample_rand::Context &memory, CUstream);

        void sample_dev(sample_rand::Context &memory, CUstream);

    };
}

#endif //SORTS_SAMPLE_RAND_DEVICE_H
