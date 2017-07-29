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

#define THREADS_IN_BLOCK 1024


//#define PRINT(...) fprintf(stderr, ##__VA_ARGS__)

#define PRINT(...) {}
#define PRINT1(...) fprintf(stderr, ##__VA_ARGS__)

using namespace std;



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

    void moveResult() {
        cuMemcpy(deviceToSort, out, sizeof(int) * size);
        cuMemsetD32(out, 0, size);
    }

public:
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


    void chujowy(Memory &memory);
    void scatter(Memory &memory);

    void counters(Memory &memory);

    void odd_even(Memory &memory);
};


Memory::Memory(int size) : size(size), sample_offsets() {

    number_of_blocks = (size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;
    y_dim = (number_of_blocks + x_dim - 1) / x_dim;

    cuMemAllocHost((void **) &sample_offsets, sizeof(int) * (S_SIZE + 1));
    sample_offsets[0] = 0;
    cuMemAlloc(&bstPtr, S_SIZE * sizeof(int));
    cuMemAlloc(&blockPrefsums, S_SIZE * number_of_blocks * sizeof(int));
    cuMemsetD32(blockPrefsums, 0, S_SIZE * number_of_blocks);

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

//    PRINT( "he size %d ?\n", size);
    deviceToSort = deviceToSort + sample_offsets[sample_nr] * sizeof(int);
    out = out + sample_offsets[sample_nr];
//    PRINT( "he?\n");
    size = sample_offsets[sample_nr + 1] - sample_offsets[sample_nr];
//    PRINT( "samp_offsets %d %d %d?\n", sample_nr, sample_offsets[sample_nr], sample_offsets[sample_nr + 1]);
    assert(size >= 0);
    number_of_blocks = (size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;


    //wydajniej
    if (size > M) {
        cuMemAllocHost((void **) &sample_offsets, sizeof(int) * (S_SIZE + 1));
    }
    sample_offsets[0] = 0;

    y_dim = x_dim == 0 ? x_dim : (number_of_blocks + x_dim - 1) / x_dim;

}

void fprintfMem(Memory &mem) {
    int * host;
//    cuMemAllocHost((void**)&host, mem.size* sizeof(int));
//    cuMemcpyDtoH(host, mem.deviceToSort, mem.size * sizeof(int));
//    PRINT( "\nMEM\n");
//    for (int i=0; i< 64 && i < mem.size; ++i) {
//        PRINT( "%d ",host[i]);
//    }
//    PRINT( "\n");
//    cuMemFreeHost(host);
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
    manageResult(cuLaunchKernel(scatterCU, memory.x_dim, memory.y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::counters(Memory &memory) {
    void *args1[] = {&memory.deviceToSort, &memory.bstPtr, &memory.blockPrefsums, &memory.number_of_blocks};
    manageResult(cuLaunchKernel(countersCU, memory.x_dim, memory.y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0),
                 "running");
    cuCtxSynchronize();
}

void Device::odd_even(Memory &memory) {
    void *args[1] = {&memory.deviceToSort};
    manageResult(cuLaunchKernel(cuOdeven, memory.x_dim, memory.y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0),
                 "running");
    cuCtxSynchronize();

}

void Device::chujowy(Memory &memory) {
    void *args[2] = {&memory.deviceToSort, &memory.size};
    manageResult(cuLaunchKernel(chujowy_sortDev, 1, 1, 1, 1, 1, 1, 0, 0, args, 0),
                 "running");
    cuCtxSynchronize();
}

void create_search_tree(Memory &memory) {
//    if (memory.size < 5000) {
    PRINT( "\nBEFSEARCH\n");
//    }
    int *tree;
    cuMemAllocHost((void **) &tree, S_SIZE * sizeof(int));
    int *sample;
    cuMemAllocHost((void **) &sample, S_SIZE * sizeof(int));
    int *to_sort;
    cuMemAllocHost((void **) &to_sort, memory.size * sizeof(int));
    cuMemcpyDtoH(to_sort, memory.deviceToSort, memory.size * sizeof(int));
//    std::copy(to_sort, to_sort + S_SIZE, sample);
    int delta = memory.size / S_SIZE;

    for (int i = (delta - 1); i < memory.size; i += delta) {
        sample[i / delta] = to_sort[memory.size - i - 1];
    }

    std::sort(sample, sample + S_SIZE);
    int iteratr = 0;
    for (int i = 2; i <= S_SIZE; i *= 2) {
        for (int j = 1; j < i; j += 2) {
            tree[iteratr++] = sample[j * (S_SIZE / i) - 1];
        }
    }
    if (memory.size < 10000) {
        for (int i = 0; i < 64; i++) {
            PRINT( "%d ", to_sort[i]);
        }
        PRINT( "\n\n\n");

        for (int i = 0; i < 64; ++i) {
            PRINT( "%d ", sample[i]);
        }
        PRINT( "\n\n\n");
//
        for (int i = 0; i < 64; ++i) {
            PRINT( "%d ", tree[i]);
        }
        PRINT( "\n");
//        assert(false);
    }


    cuMemcpyHtoD(memory.bstPtr, tree, S_SIZE * sizeof(int));
    cuMemFreeHost(tree);
    cuMemFreeHost(sample);
//    if (memory.size < 5000) {
    PRINT( "SFTER\n");
//    }
}

inline void prefsum(Memory &memory, Device &device) {
    int size = memory.number_of_blocks * S_SIZE;
    int *maxPrefSums;
    //here can be bug if size is big
    int number_of_local_blocks = (size / 2 + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;

    int x_dim = number_of_local_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_local_blocks;
    int y_dim = (number_of_local_blocks + x_dim - 1) / x_dim;

//    cout << "pref dziwne "<< x_dim << " " << y_dim<< endl;
    cuMemAllocHost((void **) &maxPrefSums, (number_of_local_blocks + 1) * sizeof(int));

    int *copy;
    cuMemAllocHost((void **) &copy, (64) * sizeof(int));
    cuMemcpyDtoH(copy, (memory.blockPrefsums + 4096 * 2), 64 * sizeof(int));

    for (int i = 0; i < 64; ++i) {
//        cout <<"copy befz "<< copy[i] <<endl;
    }

    void *args[] = {&memory.blockPrefsums, &maxPrefSums};
//    std::cout << "running" << std::endl;
    manageResult(cuLaunchKernel(device.prefsumDev, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0), "pref2");
    cuCtxSynchronize();

    maxPrefSums[0] = 0;

    cuMemcpyDtoH(copy, memory.blockPrefsums, 64 * sizeof(int));

//    cout << "after"<< endl;
//    for (int i = 0 ; i < 64; ++i) {
//        cout << copy[i] <<" ";
//    }
//    cout << endl;

//    for (int j=0; j <= number_of_local_blocks; ++j) {
//        cout<<"ps " << maxPrefSums[j] << endl;
//    }


    for (int j = 1; j <= number_of_local_blocks; ++j) {
        maxPrefSums[j] += maxPrefSums[j - 1];
    }

    void *args1[] = {&memory.blockPrefsums, &maxPrefSums, &memory.number_of_blocks, &memory.sample_offsets};

    manageResult(cuLaunchKernel(device.prefsumDev1, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0), "pref1");
    cuCtxSynchronize();

    cuMemcpyDtoH(copy, memory.blockPrefsums, 2048 * sizeof(int));

//    cout << "copy"<< endl;
//    for (int j=0; j < 64; ++j) {
//        cout <<copy[j] <<" ";
//    }
//    cout << endl;
//    for (int j=0; j< memory.number_of_blocks; ++j) {
//        cout<< memory.sample_offsets[j]<< " ";
//    }
//    cout << endl;

    cuMemFreeHost(maxPrefSums);
//    exit(1);
}

int counter = 0;

void sample_rand(Device &device, Memory &memory) {
    assert(counter++ < 1028);
//    PRINT( "search?\n");
    create_search_tree(memory);
//    PRINT( "counters?\n");
    device.counters(memory);
//    PRINT( "prefsum?\n");
    prefsum(memory, device);
//    PRINT( "scatterr?\n");
    device.scatter(memory);
//    PRINT( "no elo\n");
    memory.moveResult();
    bool all_too_small = true;
    // could be more efficient
    int sizebef = memory.size;
    PRINT( "father %d\n", sizebef);
    for (int i = 0; i < S_SIZE; ++i) {
        int offset = memory.sample_offsets[i];
//        PRINT( "offset %d\n", offset);
        int size = memory.sample_offsets[i + 1] - memory.sample_offsets[i];
        if (size > 0) {
            if (memory.size > 4000 && memory.size < 5000) {
                PRINT( "%d ", i);
            }
            if (memory.size > 5000) {
//                PRINT( "\nBIG %d\n", i);
            }
            Memory mem(memory, i);

            assert(memory.size == sizebef);
            if (mem.size == memory.size) {
                PRINT( "%d %d\n", i, memory.sample_offsets[i]);
//                int *bst;
//                cuMemAllocHost((void **) &bst, mem.size * sizeof(int));
//                cuMemcpyDtoH(bst, memory.bstPtr, S_SIZE * sizeof(int));
//                for (int i=1000 ; i< 1024; ++i) {
//                    PRINT( "%d ", bst[i]);
//                }
//                PRINT( "\n");
//                int *to_sort;
//                cuMemAllocHost((void **) &to_sort, mem.size * sizeof(int));
//                cuMemcpyDtoH(to_sort, mem.deviceToSort, mem.size * sizeof(int));
//                for (int i=0 ; i< 64; ++i) {
//                    PRINT( "%d ", to_sort[i]);
//                }
                PRINT( "\n");
                assert(false);
            }
//            PRINT( "herez? %d %d %d\n",i, memory.size, mem.size);
            //could be more efficient
            if (mem.size > M) {

                all_too_small = false;
                sample_rand(device, mem);
            } else {
//                PRINT( "bef\n");
                fprintfMem(mem);
                device.chujowy(mem);
                fprintfMem(mem);
//                device.odd_even(mem);
            }
            if (memory.size > 5000 && mem.size == 1) {
                int *bst;
                cuMemAllocHost((void **) &bst, S_SIZE * memory.number_of_blocks * sizeof(int));
                cuMemcpyDtoH(bst, memory.blockPrefsums, S_SIZE * memory.number_of_blocks * sizeof(int));
                PRINT1( "\n");
                for (int i = 1; i < memory.size; ++i) {
                    if (bst[i] > bst[i - 1]) {
                        PRINT1( "(%d %d) ", i, bst[i]);
                    }
                }
                int *bst1;
                cuMemAllocHost((void **) &bst1, S_SIZE * sizeof(int));
                cuMemcpyDtoH(bst1, memory.bstPtr, S_SIZE * sizeof(int));
                PRINT1( "\n");
                for (int i = 0; i < S_SIZE; ++i) {
                    if (bst1[i] < 30) {
                        PRINT1( "b(%d %d) ", i, bst1[i]);
                    }
                }
                PRINT1( "\n");


                PRINT1( "\nWAT? %d %d\n", i, memory.sample_offsets[i + 1]);
                assert(false);
            }
        }

    }
    if (memory.size > M) {
        cuMemFreeHost(memory.sample_offsets);
        if (memory.size < 5000) PRINT( "TUTAJ?");
    }
    if (all_too_small) return;
}


void sample_rand(int *to_sort, int size) {
    Device device;
    Memory memory(size);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    cuMemcpyHtoD(memory.deviceToSort, to_sort, size * sizeof(int));

    PRINT("beforedsa\n");
    sample_rand(device, memory);
    PRINT( "after\n");

    cuMemcpyDtoH((void *) to_sort, memory.deviceToSort, size * sizeof(int));

    memory.clean();
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(device.cuContext);

}

