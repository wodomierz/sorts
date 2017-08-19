//
// Created by Miron Ficak on 09/07/2017.
//
#include "sample_rand.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"
#include "sample_rand_context.h"
#include "sample_rand_device.h"
#include "../quick-sort/quick_sort_device.h"
#include <algorithm>
#include <assert.h>

using namespace std;

inline void prefsum(sample_rand::Context &memory, sample_rand::Device &device) {
    sample_rand::PrefsumContext prefsumMemory(memory.prefsumSize());
    device.localPrefSums(memory, prefsumMemory);

    device.prefsumOfBatchSums(prefsumMemory);


    device.globalPrefSums(memory, prefsumMemory);
    prefsumMemory.clean();
}


void sampleRand(sample_rand::Device &device, quick::Device &quickDevice, sample_rand::Context &memory) {
    if (memory.baseData.size <= M) {
        DevArray *seqs = cuMemAllocH<DevArray>(1);
        seqs[0] = DevArray(0, memory.baseData.size);
        quickDevice.lqsort(seqs, 1, memory.deviceToSort, memory.out);
        memory.moveResult();
        cuMemFreeHost(seqs);
//        device.chujowy();
        return;
    }
    device.sample_dev(memory);
    device.counters(memory);
    prefsum(memory, device);
    device.scatter(memory);
    memory.moveResult();
    // could be more efficient
    for (int i = 0; i < S_SIZE; ++i) {
        int size = memory.sample_offsets[i + 1] - memory.sample_offsets[i];
        if (size > 1) {
            sample_rand::Context mem(memory, i);
            //could be more efficient
            sampleRand(device, quickDevice, mem);
        }

    }
    memory.localClean();
}


void sampleRand(int *to_sort, int size) {
    int cudaVersion;
    cuDriverGetVersion(&cudaVersion);
    sample_rand::Device device;
    quick::Device quick_device;
    sample_rand::Context memory(size);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    cuMemcpyHtoD(memory.deviceToSort, to_sort, size * sizeof(int));

    PRINT1("beforedsa %d\n", size);
    sampleRand(device, quick_device, memory);
    PRINT("after\n");

    cuMemcpyDtoH((void *) to_sort, memory.deviceToSort, size * sizeof(int));

    memory.clean();
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(device.cuContext);

}

