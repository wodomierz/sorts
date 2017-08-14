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
        CUfunction prefsumDev1;
        CUfunction countersCU;
        CUfunction scatterCU;

        CUfunction chujowy_sortDev;


        CUfunction sampleDev;

        Device();

        void localPrefSums(sample_rand::Context &context, sample_rand::PrefsumContext &prefsumMemory);

        void prefsumOfBatchSums(sample_rand::PrefsumContext &prefsumMemory);

        void globalPrefSums(sample_rand::Context &memory, sample_rand::PrefsumContext &prefsumMemory);

        void chujowy(sample_rand::Context &memory);

        void scatter(sample_rand::Context &memory);

        void counters(sample_rand::Context &memory);

        void odd_even(sample_rand::Context &memory);

        void sample_dev(sample_rand::Context &memory);

    };
}

#endif //SORTS_SAMPLE_RAND_DEVICE_H
