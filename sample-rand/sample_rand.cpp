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

inline void prefsum(sample_rand::Context &memory, sample_rand::Device &device, CUstream cUstream) {
    sample_rand::PrefsumContext prefsumMemory(memory.prefsumSize(), memory.prefsumsMem);
    device.localPrefSums(memory, prefsumMemory, cUstream);
    device.prefsumOfBatchSums(prefsumMemory, cUstream);
    device.globalPrefSums(memory, prefsumMemory, cUstream);
}




void sample_sort_big_work_unit(sample_rand::Device &device, sample_rand::Context &memory, CUstream stream) {
    device.sample_dev(memory, stream);
    device.counters(memory, stream);
    prefsum(memory, device, stream);
    device.scatter(memory, stream);
}


void sampleRand(sample_rand::Device &device, quick::Device &quickDevice, sample_rand::Context &globalContext) {
    std::vector<DevArray> smallSize;
    std::vector<sample_rand::Context> bigSize;
    std::vector<sample_rand::Context> news;
    bigSize.push_back(globalContext);

    long totalTim = 0;


    CUstream streams[128];
    for (int i=0;i<128; ++i) {
        cuStreamCreate(streams + i, 0);
    }

    while(!bigSize.empty()) {
        cuMemsetD32(globalContext.blockPrefsums, 0, globalContext.allPrefsumsCapacity());

        for (int i=0; i< bigSize.size(); ++i) {
            std::clock_t start = std::clock();
            sample_sort_big_work_unit(device, bigSize[i], streams[i%128]);
            long delta = (std::clock() - start) / 1000;
        }
        cuCtxSynchronize();
        globalContext.moveResult(); //???
        int prefsum_offset = 0;
        int prefMemOff = 0;
        for (sample_rand::Context& workUnit: bigSize) {
            for (int i=0; i < S_SIZE; i++) {
                DevArray devArray = {workUnit.sample_offsets[i] + workUnit.offset, workUnit.sample_offsets[i+1]+ workUnit.offset};
                int size = arraySize(devArray);
                if (size > 1) {
                    if (size <= SAMPLE_OTHER_SORT_LIM) {
                        smallSize.push_back(devArray);
                    } else {
                        sample_rand::Context context(globalContext, devArray.start, size, prefsum_offset,prefMemOff, news.size());
                        prefsum_offset += context.prefsumSize();
                        prefMemOff += (ceil_div(context.prefsumSize(), PREFSUM_BLOCK_SIZE) + 1);
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
    DevArray* seqs = cuMemAllocH<DevArray>(smallSize.size());
    for (int i=0; i < smallSize.size(); ++i) {
        seqs[i] = smallSize[i];
    }
    quickDevice.lqsort(seqs, smallSize.size(), globalContext.deviceToSort, globalContext.out);
    cuMemFreeHost(seqs);

    for (int i=0;i<128; ++i) {
        cuStreamDestroy(streams[i]);
    }
}


void sampleRand(int *to_sort, int size) {
    int cudaVersion;
    cuDriverGetVersion(&cudaVersion);
    sample_rand::Device device;
    quick::Device quick_device;
    sample_rand::Context memory(size);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    cuMemcpyHtoD(memory.deviceToSort, to_sort, size * sizeof(int));

    sampleRand(device, quick_device, memory);

    cuMemcpyDtoH((void *) to_sort, memory.deviceToSort, size * sizeof(int));

    memory.clean();
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(device.cuContext);

}

