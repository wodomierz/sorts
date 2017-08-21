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




void sample_sort_big_work_unit(sample_rand::Device &device, sample_rand::Context &memory) {
    device.sample_dev(memory);
    device.counters(memory);
    prefsum(memory, device);
    device.scatter(memory);
//    memory.moveResult(); ??
}


void sampleRand(sample_rand::Device &device, quick::Device &quickDevice, sample_rand::Context &globalContext) {
    std::vector<DevArray> smallSize;
    std::vector<sample_rand::Context> bigSize;
    std::vector<sample_rand::Context> news;
    bigSize.push_back(globalContext);
    while(!bigSize.empty()) {
        cuMemsetD32(globalContext.blockPrefsums, 0, globalContext.allPrefsumsCapacity());
        for (sample_rand::Context& workUnit: bigSize) {
            sample_sort_big_work_unit(device, workUnit);
        }
        //move result here or in kernel??
        globalContext.moveResult(); //???
//        cuCtxSynchronize(); //to obtain offsets

        //async???
        int prefsum_offset = 0;
        for (sample_rand::Context& workUnit: bigSize) {
            for (int i=0; i < S_SIZE; i++) {
                DevArray devArray = {workUnit.sample_offsets[i] + workUnit.offset, workUnit.sample_offsets[i+1]+ workUnit.offset};
                int size = arraySize(devArray);
                if (size > 1) {
                    if (size <= SAMPLE_OTHER_SORT_LIM) {
                        smallSize.push_back(devArray);
                    } else {
                        sample_rand::Context context(globalContext, devArray.start, size, prefsum_offset, news.size());
                        prefsum_offset += context.prefsumSize();
                        news.push_back(context);
                    }
                }
            }
        }
        bigSize.clear();
        bigSize.insert(bigSize.end(), news.begin(), news.end());
        news.clear();
    }
    cuCtxSynchronize();
    //async?
    DevArray* seqs = cuMemAllocH<DevArray>(smallSize.size());
    for (int i=0; i < smallSize.size(); ++i) {
        seqs[i] = smallSize[i];
    }
    quickDevice.lqsort(seqs, smallSize.size(), globalContext.deviceToSort, globalContext.out);
    cuMemFreeHost(seqs);

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

