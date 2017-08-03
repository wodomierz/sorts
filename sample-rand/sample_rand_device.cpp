//
// Created by Miron Ficak on 03/08/2017.
//

#include <cassert>
#include "sample_rand_device.h"
#include "sample_rand.h"

namespace sample_rand {

    Device::Device() {
        cuInit(0);
        manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
        manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
        manageResult(cuModuleLoad(&cuModule, "sample-rand/sample_rand.ptx"), "cannot load module");
        manageResult(cuModuleGetFunction(&prefsumDev, cuModule, "prefsum"), "cannot load function");
        manageResult(cuModuleGetFunction(&prefsumDev1, cuModule, "prefsum1"), "cannot load function");
        manageResult(cuModuleGetFunction(&countersCU, cuModule, "counters"), "cannot load function");
        manageResult(cuModuleGetFunction(&scatterCU, cuModule, "scatter"), "cannot load function");

        manageResult(cuModuleGetFunction(&chujowy_sortDev, cuModule, "chujowy_sort"), "cannot load function");

        manageResult(cuModuleGetFunction(&cuOdeven, cuModule, "odd_even"), "cannot load function");

    }

    void Device::scatter(sample_rand::Context &memory) {
        BaseData& baseData = memory.baseData;
        void *args2[]{&memory.deviceToSort, &memory.out, &memory.bstPtr, &memory.blockPrefsums, &baseData.number_of_blocks,&memory.baseData.size};
        manageResult(cuLaunchKernel(scatterCU, baseData.x_dim, baseData.y_dim, 1, T, 1, 1, 0, 0, args2, 0),
                     "running");
        cuCtxSynchronize();
    }

    void Device::counters(sample_rand::Context &memory) {
        BaseData& baseData = memory.baseData;
        void *args1[] = {&memory.deviceToSort, &memory.bstPtr, &memory.blockPrefsums, &memory.baseData.number_of_blocks, &memory.baseData.size};
        manageResult(cuLaunchKernel(countersCU, baseData.x_dim, baseData.y_dim, 1, T, 1, 1, 0, 0, args1, 0),
                     "running");
        cuCtxSynchronize();
    }

    void Device::odd_even(sample_rand::Context &memory) {
        void *args[1] = {&memory.deviceToSort};
        manageResult(cuLaunchKernel(cuOdeven, memory.baseData.size / 2, 1, 1, M / 2, 1, 1, 0, 0, args, 0),
                     "running");
        cuCtxSynchronize();

    }

    void Device::chujowy(sample_rand::Context &memory) {
//    assert(false);
        assertPrintable([memory]{PRINT1("%d %d\n", memory.baseData.size, M);}, memory.baseData.size == BLOCK_SIZE);
        void *args[2] = {&memory.deviceToSort, &memory.baseData.size};
        manageResult(cuLaunchKernel(chujowy_sortDev, 1, 1, 1, 1, 1, 1, 0, 0, args, 0),
                     "running");
        cuCtxSynchronize();
    }

    void Device::localPrefSums(sample_rand::Context &memory, sample_rand::PrefsumContext &prefsumMemory) {
        void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums, &prefsumMemory.baseData.size};

        manageResult(cuLaunchKernel(prefsumDev, prefsumMemory.baseData.x_dim, prefsumMemory.baseData.y_dim, 1, PREFSUM_THREADS, 1, 1, 0, 0, args, 0),
                     "pref");
        cuCtxSynchronize();
    }

    void Device::globalPrefSums(sample_rand::Context &memory, sample_rand::PrefsumContext &prefsumMemory) {
        void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums, &memory.baseData.number_of_blocks, &memory.sample_offsets,&prefsumMemory.baseData.size};
        manageResult(cuLaunchKernel(prefsumDev1, prefsumMemory.baseData.x_dim, prefsumMemory.baseData.y_dim, 1, PREFSUM_THREADS, 1, 1, 0, 0, args, 0),
                     "pref1");
        cuCtxSynchronize();
    }

    void Device::prefsumOfBatchSums(sample_rand::PrefsumContext &prefsumMemory) {
        //to jest kurwa wolne
        prefsumMemory.batchSums[0] = 0;
        for (int j = 1; j <= prefsumMemory.baseData.number_of_blocks; ++j) {
            prefsumMemory.batchSums[j] += prefsumMemory.batchSums[j - 1];
        }
    }


}