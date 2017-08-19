//
// Created by Miron Ficak on 03/08/2017.
//

#include <cassert>
#include "sample_rand_device.h"
#include "sample_rand.h"
#include "../quick-sort/quick_debug.h"

namespace sample_rand {

    Device::Device() {
        cuInit(0);
        manageResult(cuDeviceGet(&cuDevice, 0));
        manageResult(cuCtxCreate(&cuContext, 0, cuDevice));
        manageResult(cuModuleLoad(&cuModule, "sample-rand/sample_rand.ptx"));
        manageResult(cuModuleGetFunction(&prefsumDev, cuModule, "prefsum"));
        manageResult(cuModuleGetFunction(&prefsumDev1, cuModule, "prefsum1"));
        manageResult(cuModuleGetFunction(&countersCU, cuModule, "counters"));
        manageResult(cuModuleGetFunction(&scatterCU, cuModule, "scatter"));

        manageResult(cuModuleGetFunction(&chujowy_sortDev, cuModule, "chujowy_sort"));
        manageResult(cuModuleGetFunction(&sampleDev, cuModule, "sample"));

    }

    void Device::scatter(sample_rand::Context &memory) {
        BaseData &baseData = memory.baseData;
        void *args2[]{&memory.deviceToSort, &memory.out, &memory.bstPtr, &memory.blockPrefsums,
                      &baseData.number_of_blocks, &memory.baseData.size};
        safeLaunch1Dim(scatterCU, baseData.x_dim, baseData.y_dim, THREADS_PER_BLOCK, args2);
        cuCtxSynchronize();
    }

    void Device::counters(sample_rand::Context &memory) {
        BaseData &baseData = memory.baseData;
        void *args1[] = {&memory.deviceToSort, &memory.bstPtr, &memory.blockPrefsums, &memory.baseData.number_of_blocks,
                         &memory.baseData.size};
        safeLaunch1Dim(countersCU, baseData.x_dim, baseData.y_dim, THREADS_PER_BLOCK, args1);
        cuCtxSynchronize();
    }

    void Device::chujowy(sample_rand::Context &memory) {
        void *args[2] = {&memory.deviceToSort, &memory.baseData.size};
        safeLaunch1Dim(chujowy_sortDev, 1, 1, 1, args);
        cuCtxSynchronize();
    }

    void Device::localPrefSums(sample_rand::Context &memory, sample_rand::PrefsumContext &prefsumMemory) {
        void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums, &prefsumMemory.baseData.size};

        safeLaunch1Dim(prefsumDev, prefsumMemory.baseData.x_dim, prefsumMemory.baseData.y_dim, PREFSUM_THREADS, args);
        cuCtxSynchronize();
    }

    void Device::globalPrefSums(sample_rand::Context &memory, sample_rand::PrefsumContext &prefsumMemory) {
        void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums, &memory.baseData.number_of_blocks,
                        &memory.sample_offsets, &prefsumMemory.baseData.size};
        safeLaunch1Dim(prefsumDev1, prefsumMemory.baseData.x_dim, prefsumMemory.baseData.y_dim, PREFSUM_THREADS, args);
        cuCtxSynchronize();
    }

    void Device::prefsumOfBatchSums(sample_rand::PrefsumContext &prefsumMemory) {
        //to jest kurwa wolne
        prefsumMemory.batchSums[0] = 0;
        for (int j = 1; j <= prefsumMemory.baseData.number_of_blocks; ++j) {
            prefsumMemory.batchSums[j] += prefsumMemory.batchSums[j - 1];
        }
    }

    void Device::sample_dev(sample_rand::Context &memory) {
        int plus = rand() % 100000;
        int seed = rand() % 100000;
        void *args[] = {&memory.deviceToSort, &memory.baseData.size, &seed, &plus, &memory.bstPtr};
        safeLaunch1Dim(sampleDev, 1, 1, SAMPLE_THREADS, args);
        cuCtxSynchronize();
    }


}