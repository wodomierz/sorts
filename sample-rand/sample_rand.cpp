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

class PrefsumMemory {
public:
    int *batchSums;
    int size;
    int x_dim;
    int y_dim;
    int number_of_prefsum_blocks;

    void clean();

    PrefsumMemory(int);

};

PrefsumMemory::PrefsumMemory(int size) : size(size) {
    //here can be bug if size is big
    number_of_prefsum_blocks = (size / 2 + T - 1) / T;
    x_dim = number_of_prefsum_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_prefsum_blocks;
    y_dim = (number_of_prefsum_blocks + x_dim - 1) / x_dim;
    cuMemAllocHost((void **) &batchSums, (number_of_prefsum_blocks + 1) * sizeof(int));
}

void PrefsumMemory::clean() {
    cuMemFreeHost(batchSums);
}


class Memory {
public:
    int *sample_offsets;
    int size;
    int number_of_blocks;
    int x_dim;
    int y_dim;
    CUdeviceptr blockPrefsums;
    CUdeviceptr deviceToSort;
    CUdeviceptr out;
    CUdeviceptr bstPtr;


    int prefsumSize();

    void moveResult() {
        cuMemcpy(deviceToSort, out, sizeof(int) * size);
        cuMemsetD32(out, 0, size);
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


Memory::Memory(int size) : size(size), sample_offsets() {

    number_of_blocks = (size + T - 1) / T;
    x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;
    y_dim = (number_of_blocks + x_dim - 1) / x_dim;

    sample_offsets = 0;
    cuMemAllocHost((void **) &sample_offsets, sizeof(int) * (S_SIZE + 1));
    sample_offsets[0] = 0;
    cuMemAlloc(&bstPtr, S_SIZE * sizeof(int));
    cuMemAlloc(&blockPrefsums, prefsumSize() * sizeof(int));
    cuMemsetD32(blockPrefsums, 0, prefsumSize());

    cuMemAlloc(&deviceToSort, size * sizeof(int));
    cuMemAlloc(&out, size * sizeof(int));
    cuMemsetD32(out, 0, size);


}

void Memory::clean() {

    cuMemFree(deviceToSort);
    cuMemFree(out);
    cuMemFree(blockPrefsums);
    cuMemFree(bstPtr);
}

Memory::Memory(Memory &memory, int sample_nr) : Memory(memory) {
    deviceToSort = deviceToSort + sample_offsets[sample_nr] * sizeof(int);
    out = out + sample_offsets[sample_nr] * sizeof(int);
    size = sample_offsets[sample_nr + 1] - sample_offsets[sample_nr];
    assert(size >= 0);
    number_of_blocks = (size + T - 1) / T;
    x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;
    //wydajniej
    if (size > M) {
        cuMemAllocHost((void **) &sample_offsets, sizeof(int) * (S_SIZE + 1));
        sample_offsets[0] = 0;
    }
    y_dim = x_dim == 0 ? x_dim : (number_of_blocks + x_dim - 1) / x_dim;
}

int Memory::prefsumSize() {
    return number_of_blocks * S_SIZE;
}

void Memory::localClean() {
    if (size > M) {
        cuMemFreeHost(sample_offsets);
    }

}

void fprintfMem(Memory &mem) {
    PRINT1("\nMEM %d\n", mem.size);
    print_Devtab(mem.deviceToSort, mem.size, 64);
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
    void *args2[]{&memory.deviceToSort, &memory.out, &memory.bstPtr, &memory.blockPrefsums, &memory.number_of_blocks};
    manageResult(cuLaunchKernel(scatterCU, memory.x_dim, memory.y_dim, 1, T, 1, 1, 0, 0, args2, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::counters(Memory &memory) {
//    if (memory.size == 128) {
//        PRINT1("x %d y %d size %d blocks %d  threads %d", memory.x_dim, memory.y_dim, memory.size, memory.number_of_blocks, T);
//    }
    void *args1[] = {&memory.deviceToSort, &memory.bstPtr, &memory.blockPrefsums, &memory.number_of_blocks};
    manageResult(cuLaunchKernel(countersCU, memory.x_dim, memory.y_dim, 1, T, 1, 1, 0, 0, args1, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::odd_even(Memory &memory) {
    void *args[1] = {&memory.deviceToSort};
    manageResult(cuLaunchKernel(cuOdeven, memory.size / 2, 1, 1, M / 2, 1, 1, 0, 0, args, 0),
                 "running");
    cuCtxSynchronize();

}

void Device::chujowy(Memory &memory) {
    assertPrintable([memory]{PRINT1("%d %d\n", memory.size, M);}, memory.size == 4);
    void *args[2] = {&memory.deviceToSort, &memory.size};
    manageResult(cuLaunchKernel(chujowy_sortDev, 1, 1, 1, 1, 1, 1, 0, 0, args, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::localPrefSums(Memory &memory, PrefsumMemory &prefsumMemory) {
    void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums};
    manageResult(cuLaunchKernel(prefsumDev, prefsumMemory.x_dim, prefsumMemory.y_dim, 1, T, 1, 1, 0, 0, args, 0),
                 "pref");
    cuCtxSynchronize();
}

void Device::globalPrefSums(Memory &memory, PrefsumMemory &prefsumMemory) {
    void *args[] = {&memory.blockPrefsums, &prefsumMemory.batchSums, &memory.number_of_blocks, &memory.sample_offsets};
    manageResult(cuLaunchKernel(prefsumDev1, prefsumMemory.x_dim, prefsumMemory.y_dim, 1, T, 1, 1, 0, 0, args, 0),
                 "pref1");
    cuCtxSynchronize();
}

void Device::prefsumOfBatchSums(PrefsumMemory &prefsumMemory) {
    //to jest kurwa wolne
    prefsumMemory.batchSums[0] = 0;
    for (int j = 1; j <= prefsumMemory.number_of_prefsum_blocks; ++j) {
        prefsumMemory.batchSums[j] += prefsumMemory.batchSums[j - 1];
    }
}

void create_search_tree(Memory &memory) {
    //WORKS ONLY IF memory.size > S_SIZE
    int *tree;
    cuMemAllocHost((void **) &tree, S_SIZE * sizeof(int));
    int *sample;
    cuMemAllocHost((void **) &sample, S_SIZE * sizeof(int));
    int *to_sort;
    cuMemAllocHost((void **) &to_sort, memory.size * sizeof(int));
    cuMemcpyDtoH(to_sort, memory.deviceToSort, memory.size * sizeof(int));
    int delta = memory.size / S_SIZE;

    std::sort(to_sort, to_sort + memory.size);

    for (int i = (delta - 1); i < memory.size; i += delta) {
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
//    if (memory.size == 128) {
//        print_Devtab(memory.deviceToSort, memory.size, memory.size, 0, "BEFORE");
//    }
    create_search_tree(memory);

//    if (memory.size == 128) {
//        print_Devtab(memory.blockPrefsums, memory.prefsumSize(), memory.size, 0, "COUNTERS BEF");
//    }
    device.counters(memory);
    if (memory.size == 128) {
//        print_Devtab(memory.blockPrefsums, memory.prefsumSize(), memory.size, 0, "COUNTERS");
    }
    prefsum(memory, device);
    if (memory.size == 128) {
//        print_Devtab(memory.blockPrefsums, memory.prefsumSize(), memory.size, 0, "PREFSUMS");
    }
    device.scatter(memory);
    memory.moveResult();

    if (memory.size == 128) {
//        print_Devtab(memory.deviceToSort, memory.size, memory.size, 0, "SCATTER");
    }

    // could be more efficient
//    PRINT1("father %d\n", memory.size);
    for (int i = 0; i < S_SIZE; ++i) {
        int offset = memory.sample_offsets[i];
//        PRINT1( "o %d f %d\n", offset, memory.size);
        int size = memory.sample_offsets[i + 1] - memory.sample_offsets[i];
//        PRINT1( "o1 %d f %d\n", memory.sample_offsets[i + 1], memory.size);
        if (size > 0) {
            Memory mem(memory, i);
            assertPrintable(([i, memory, mem] {
                PRINT1("%d %d %d %d %d\n",
                       i,
                       memory.sample_offsets[i],
                       memory.sample_offsets[i + 1],
                       memory.size,
                       mem.size);
            }),
                            memory.size != mem.size);
            //could be more efficient
            if (mem.size > M) {
                sample_rand(device, mem);
            } else {
                device.chujowy(mem);
            }
            if (memory.size > 5000 && mem.size == 1) {
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
                print_Devtab(memory.deviceToSort, memory.size, 512);
                PRINT1("\nWAT? %d %d %d %d\n", i, memory.size, memory.sample_offsets[i], memory.sample_offsets[i + 1]);
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

