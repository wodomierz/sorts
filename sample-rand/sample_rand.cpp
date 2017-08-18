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


void sampleRand(sample_rand::Device &device, sample_rand::Context &memory) {
    device.sample_dev(memory);
    device.counters(memory);
    prefsum(memory, device);
    device.scatter(memory);
    memory.moveResult();
    // could be more efficient
//    PRINT1("father %d\n", memory.baseData.size);
    for (int i = 0; i < S_SIZE; ++i) {
        int offset = memory.sample_offsets[i];
//        PRINT1( "o %d f %d\n", offset, memory.baseData.size);
        int size = memory.sample_offsets[i + 1] - memory.sample_offsets[i];
//        PRINT1( "o1 %d f %d\n", memory.sample_offsets[i + 1], memory.baseData.size);
        if (size > 1) {
            sample_rand::Context mem(memory, i);
            //could be more efficient
            if (mem.baseData.size > M) {
                sampleRand(device, mem);
            } else {
                device.chujowy(mem);
            }
        }

    }
    memory.localClean();
}


void sampleRand(int *to_sort, int size) {
    int cudaVersion;
    cuDriverGetVersion(&cudaVersion);
    sample_rand::Device device;
    sample_rand::Context memory(size);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    cuMemcpyHtoD(memory.deviceToSort, to_sort, size * sizeof(int));

    PRINT1("beforedsa %d\n", size);
    sampleRand(device, memory);
    PRINT("after\n");

    cuMemcpyDtoH((void *) to_sort, memory.deviceToSort, size * sizeof(int));

    memory.clean();
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(device.cuContext);

}

