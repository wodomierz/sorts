//
// Created by Miron Ficak on 09/07/2017.
//
#include "sample_rand.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"
#include <algorithm>
#include <assert.h>
#include <cuda_runtime_api.h>

using namespace std;

struct BaseData {
    int x_dim;
    int y_dim;
    int number_of_blocks;
    int size;
    BaseData(int size, int block_size): size(size) {
        number_of_blocks = ceil_div(size, block_size);
        x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;
        y_dim = ceil_div(number_of_blocks, x_dim);
    }
};

class PrefsumMemory {
public:
    BaseData baseData;
    int *batchSums;

    void clean();

    PrefsumMemory(int);

};

PrefsumMemory::PrefsumMemory(int size) : baseData(size, PREFSUM_BLOCK_SIZE) {
    batchSums = cuAllocHostInts(baseData.number_of_blocks + 1);
}



void PrefsumMemory::clean() {
    cuMemFreeHost(batchSums);
}


class Memory {
public:
    BaseData baseData;
    int *sample_offsets;
    CUdeviceptr blockPrefsums;
    CUdeviceptr deviceToSort;
    CUdeviceptr out;
    CUdeviceptr bstPtr;


    int prefsumSize();

    void moveResult() {
        cuMemcpy(deviceToSort, out, sizeof(int) * baseData.size);
        cuMemsetD32(out, 0, baseData.size);
    }

public:
    void localClean();

    void clean();

    Memory(int);

    Memory(Memory &memory, int);
};

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

    Device();

    void localPrefSums(Memory &memory, PrefsumMemory &prefsumMemory);

    void prefsumOfBatchSums(PrefsumMemory &prefsumMemory);

    void globalPrefSums(Memory &memory, PrefsumMemory &prefsumMemory);

    void chujowy(Memory &memory);

    void scatter(Memory &memory);

    void counters(Memory &memory);

    void odd_even(Memory &memory);

};


Memory::Memory(int size): baseData(size, BLOCK_SIZE) {

    sample_offsets = cuAllocHostInts(S_SIZE + 1);
    sample_offsets[0] = 0;
    bstPtr = cuAllocInts(S_SIZE);
    blockPrefsums = cuAllocInts(prefsumSize());
    cuMemsetD32(blockPrefsums, 0, prefsumSize());

    deviceToSort = cuAllocInts(size);
    out = cuAllocInts(size);
    cuMemsetD32(out, 0, size);


}

void Memory::clean() {

    cuMemFree(deviceToSort);
    cuMemFree(out);
    cuMemFree(blockPrefsums);
    cuMemFree(bstPtr);
}

Memory::Memory(Memory &memory, int sample_nr) : Memory(memory) {
    deviceToSort = addIntOffset(deviceToSort, sample_offsets[sample_nr]);
    out = addIntOffset(out, sample_offsets[sample_nr]);
    int size = sample_offsets[sample_nr + 1] - sample_offsets[sample_nr];
    assert(size >= 0);
    baseData = BaseData(size, BLOCK_SIZE);
    //wydajniej
    if (baseData.size > M) {
        sample_offsets = cuAllocHostInts(S_SIZE + 1);

    }
    sample_offsets[0] = 0;

}

int Memory::prefsumSize() {
    return baseData.number_of_blocks * S_SIZE;
}

void Memory::localClean() {
    if (baseData.size > M) {
        cuMemFreeHost(sample_offsets);
    }

}

void fprintfMem(Memory &mem) {
    PRINT1("\nMEM %d\n", mem.baseData.size);
    print_Devtab(mem.deviceToSort, mem.baseData.size, 64);
}

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

void Device::scatter(Memory &memory) {
    BaseData& baseData = memory.baseData;
    void *args2[]{&memory.deviceToSort, &memory.out, &memory.bstPtr, &memory.blockPrefsums, &baseData.number_of_blocks,&memory.baseData.size};
    manageResult(cuLaunchKernel(scatterCU, baseData.x_dim, baseData.y_dim, 1, T, 1, 1, 0, 0, args2, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::counters(Memory &memory) {
    BaseData& baseData = memory.baseData;
    void *args1[] = {&memory.deviceToSort, &memory.bstPtr, &memory.blockPrefsums, &memory.baseData.number_of_blocks, &memory.baseData.size};
    manageResult(cuLaunchKernel(countersCU, baseData.x_dim, baseData.y_dim, 1, T, 1, 1, 0, 0, args1, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::odd_even(Memory &memory) {
    void *args[1] = {&memory.deviceToSort};
    manageResult(cuLaunchKernel(cuOdeven, memory.baseData.size / 2, 1, 1, M / 2, 1, 1, 0, 0, args, 0),
                 "running");
    cuCtxSynchronize();

}

void Device::chujowy(Memory &memory) {
//    assert(false);
    assertPrintable([memory]{PRINT1("%d %d\n", memory.baseData.size, M);}, memory.baseData.size == BLOCK_SIZE);
    void *args[2] = {&memory.deviceToSort, &memory.baseData.size};
    manageResult(cuLaunchKernel(chujowy_sortDev, 1, 1, 1, 1, 1, 1, 0, 0, args, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::localPrefSums(Memory &memory, PrefsumMemory &prefsumMemory) {
    void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums, &prefsumMemory.baseData.size};

    manageResult(cuLaunchKernel(prefsumDev, prefsumMemory.baseData.x_dim, prefsumMemory.baseData.y_dim, 1, PREFSUM_THREADS, 1, 1, 0, 0, args, 0),
                 "pref");
    cuCtxSynchronize();
}

void Device::globalPrefSums(Memory &memory, PrefsumMemory &prefsumMemory) {
    void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums, &memory.baseData.number_of_blocks, &memory.sample_offsets,&prefsumMemory.baseData.size};
    manageResult(cuLaunchKernel(prefsumDev1, prefsumMemory.baseData.x_dim, prefsumMemory.baseData.y_dim, 1, PREFSUM_THREADS, 1, 1, 0, 0, args, 0),
                 "pref1");
    cuCtxSynchronize();
}

void Device::prefsumOfBatchSums(PrefsumMemory &prefsumMemory) {
    //to jest kurwa wolne
    prefsumMemory.batchSums[0] = 0;
    for (int j = 1; j <= prefsumMemory.baseData.number_of_blocks; ++j) {
        prefsumMemory.batchSums[j] += prefsumMemory.batchSums[j - 1];
    }
}

void create_search_tree(Memory &memory) {
    //WORKS ONLY IF memory.size > S_SIZE
    int *tree = cuAllocHostInts(S_SIZE);
    int *sample = cuAllocHostInts(S_SIZE);
    int *to_sort = cuAllocHostInts(memory.baseData.size);
    cuMemcpyDtoH(to_sort, memory.deviceToSort, memory.baseData.size * sizeof(int));
    int delta = memory.baseData.size / S_SIZE;

    std::sort(to_sort, to_sort + memory.baseData.size);

    for (int i = (delta - 1); i < memory.baseData.size; i += delta) {
        sample[i / delta] = to_sort[i];
    }

    std::sort(sample, sample + S_SIZE);
    int iteratr = 0;
    for (int i = 2; i <= S_SIZE; i *= 2) {
        for (int j = 1; j < i; j += 2) {
            tree[iteratr++] = sample[j * (S_SIZE / i) - 1];
        }
    }
    tree[S_SIZE - 1] = 0;
    cuMemcpyHtoD(memory.bstPtr, tree, S_SIZE * sizeof(int));
    cuMemFreeHost(tree);
    cuMemFreeHost(sample);
}

inline void prefsum(Memory &memory, Device &device) {
    PrefsumMemory prefsumMemory(memory.prefsumSize());
    device.localPrefSums(memory, prefsumMemory);

    device.prefsumOfBatchSums(prefsumMemory);


    device.globalPrefSums(memory, prefsumMemory);
    prefsumMemory.clean();
}

//int counter = 0;

void sample_rand(Device &device, Memory &memory) {
    create_search_tree(memory);

    device.counters(memory);

//    print_Devtab(memory.blockPrefsums, memory.prefsumSize(),memory.prefsumSize(), 0,"BEF");
    prefsum(memory, device);
//    print_Devtab(memory.blockPrefsums, memory.prefsumSize(),memory.prefsumSize(), 0,"aft");


    device.scatter(memory);
    memory.moveResult();

    // could be more efficient
//    PRINT1("father %d\n", memory.baseData.size);
    for (int i = 0; i < S_SIZE; ++i) {
        int offset = memory.sample_offsets[i];
//        PRINT1( "o %d f %d\n", offset, memory.baseData.size);
        int size = memory.sample_offsets[i + 1] - memory.sample_offsets[i];
//        PRINT1( "o1 %d f %d\n", memory.sample_offsets[i + 1], memory.baseData.size);
        if (size > 0) {
            Memory mem(memory, i);
            assertPrintable(([i, memory, mem] {
                PRINT1("%d %d %d %d %d\n",
                       i,
                       memory.sample_offsets[i],
                       memory.sample_offsets[i + 1],
                       memory.baseData.size,
                       mem.baseData.size);
            }),
                            memory.baseData.size != mem.baseData.size);
            //could be more efficient
            if (mem.baseData.size > M) {
                sample_rand(device, mem);
            } else {
                device.chujowy(mem);
            }
            if (memory.baseData.size > 5000 && mem.baseData.size == 1) {
                print_Devtab(
                        memory.blockPrefsums,
                        memory.prefsumSize(),
                        memory.prefsumSize(),
                        1,
                        "PREFSUM JUMPS",
                        indexedPrint,
                        [](int i, int *tab) -> bool { return (tab[i] > tab[i - 1]); }
                );
                print_Devtab(
                        memory.bstPtr,
                        S_SIZE,
                        S_SIZE,
                        1,
                        "BST",
                        indexedPrint
                );
                print_Devtab(memory.deviceToSort, memory.baseData.size, 512);
                PRINT1("\nWAT? %d %d %d %d\n", i, memory.baseData.size, memory.sample_offsets[i], memory.sample_offsets[i + 1]);
                assert(false);
            }
        }

    }
    memory.localClean();
}


void sample_rand(int *to_sort, int size) {
    int cudaVersion;
    cuDriverGetVersion(&cudaVersion);
    PRINT1("running version %d\n", cudaVersion);

    Device device;
    Memory memory(size);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    cuMemcpyHtoD(memory.deviceToSort, to_sort, size * sizeof(int));

    PRINT1("beforedsa %d\n", size);
    sample_rand(device, memory);
    PRINT("after\n");

    cuMemcpyDtoH((void *) to_sort, memory.deviceToSort, size * sizeof(int));

    memory.clean();
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(device.cuContext);

}

