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
#define THREADS_IN_BLOCK 1024

using namespace std;

class Memory {
public:
    int* sample_offsets;
    int size;
    int number_of_blocks;
    int x_dim;
    int y_dim;
    CUdeviceptr blockPrefsums;
    CUdeviceptr deviceToSort;
    CUdeviceptr out;
    CUdeviceptr bstPtr;
    void moveResult(){
        cuMemcpy(deviceToSort, out, sizeof(int) * size);
        cuMemsetD32(out, 0, size);
    }

public:
    void clean();
    Memory(int);
    Memory(Memory& memory, int);
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
    Device();
    void scatter(Memory& memory);
    void counters(Memory& memory);
    void odd_even(Memory& memory);
};




Memory::Memory(int size): size(size), sample_offsets() {

    number_of_blocks = (size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;
    y_dim = (number_of_blocks + x_dim - 1) / x_dim;

    cuMemAllocHost((void**) &sample_offsets, sizeof(int) * (S_SIZE + 1));
    sample_offsets[0] =0;
    cuMemAlloc(&bstPtr, S_SIZE * sizeof(int));
    cuMemAlloc(&blockPrefsums, S_SIZE * number_of_blocks * sizeof(int));
    cuMemsetD32(blockPrefsums, 0,  S_SIZE * number_of_blocks);

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

Memory::Memory(Memory &memory, int sample_nr): Memory(memory) {

    fprintf(stderr, "he size %d ?\n", size);
    deviceToSort = deviceToSort + sample_offsets[sample_nr]* sizeof(int);
    fprintf(stderr, "he?\n");
    out = out + sample_offsets[sample_nr];
    fprintf(stderr, "he?\n");
    size = sample_offsets[sample_nr + 1] - sample_offsets[sample_nr];
    number_of_blocks = (size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;
    fprintf(stderr, "he %d ?\n", sample_offsets[sample_nr + 1]);

    //wydajniej
    cuMemAllocHost((void**) &sample_offsets, sizeof(int) * (S_SIZE + 1));
    sample_offsets[0] =0;

    y_dim = (number_of_blocks + x_dim - 1) / x_dim;

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

    manageResult(cuModuleGetFunction(&cuOdeven, cuModule, "odd_even"), "cannot load function");

}

void Device::scatter(Memory& memory) {
    void *args2[] {&memory.deviceToSort, &memory.out, &memory.bstPtr, &memory.blockPrefsums, &memory.number_of_blocks};
    manageResult(cuLaunchKernel(scatterCU, memory.x_dim, memory.y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0), "running");
    cuCtxSynchronize();
}

void Device::counters(Memory& memory) {
    void *args1[] = {&memory.deviceToSort, &memory.bstPtr, &memory.blockPrefsums, &memory.number_of_blocks};
    manageResult(cuLaunchKernel(countersCU, memory.x_dim, memory.y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0), "running");
    cuCtxSynchronize();
}

void Device::odd_even(Memory &memory) {
    void* args[1] = {&memory.deviceToSort};
    manageResult(cuLaunchKernel(cuOdeven, memory.x_dim, memory.y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0),"running");
    cuCtxSynchronize();

}

void create_search_tree(Memory& memory) {

    int *tree;
    cuMemAllocHost((void **) &tree, S_SIZE* sizeof(int));
    int *sample;
    cuMemAllocHost((void **) &sample, S_SIZE * sizeof(int));
    int* to_sort;
    cuMemAllocHost((void **) &to_sort, memory.size* sizeof(int));
    cuMemcpyDtoH(to_sort, memory.deviceToSort, memory.size*sizeof(int));
//    std::copy(to_sort, to_sort + S_SIZE, sample);
    int delta = memory.size / S_SIZE;

    for (int i=(delta -1); i < memory.size; i+=delta) {
        sample[i/delta] = i;
    }

    std::sort(sample, sample + S_SIZE);
    int iteratr = 0;
    for (int i = 2; i <= S_SIZE; i *= 2) {
        for (int j = 1; j < i; j += 2) {
            tree[iteratr++] = sample[j * (S_SIZE / i) -1];
        }
    }
//    for (int i=0; i< S_SIZE; ++i) {
//        fprintf(stderr,"sample(%d %d)", i, sample[i]);
//    }
//    fprintf(stderr, "\n\n\n");

//    for (int i=0; i< S_SIZE; ++i) {
//        fprintf(stderr,"%d ", tree[i]);
//    }
//    fprintf(stderr, "\n");

    cuMemcpyHtoD(memory.bstPtr, tree, S_SIZE * sizeof(int));
    cuMemFreeHost(tree);
    cuMemFreeHost(sample);
}

inline void prefsum(Memory& memory, Device& device) {
    int size = memory.number_of_blocks * S_SIZE;
    int* maxPrefSums;
    //here can be bug if size is big
    int number_of_local_blocks = (size/2 + THREADS_IN_BLOCK -1)/ THREADS_IN_BLOCK;

    int x_dim = number_of_local_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_local_blocks;
    int y_dim = (number_of_local_blocks + x_dim - 1) / x_dim;

    cout << "pref dziwne "<< x_dim << " " << y_dim<< endl;
    cuMemAllocHost((void**)&maxPrefSums, (number_of_local_blocks + 1) * sizeof(int));

    int* copy;
    cuMemAllocHost((void**)&copy, (64) * sizeof(int));
    cuMemcpyDtoH(copy, (memory.blockPrefsums + 4096*2), 64 * sizeof(int));

    for (int i = 0 ; i < 64; ++i) {
//        cout <<"copy befz "<< copy[i] <<endl;
    }

    void* args[] = {&memory.blockPrefsums, &maxPrefSums };
    std::cout << "running"<< std::endl;
    manageResult(cuLaunchKernel(device.prefsumDev,x_dim, y_dim, 1,THREADS_IN_BLOCK, 1, 1, 0,0, args, 0), "pref2");
    cuCtxSynchronize();

    maxPrefSums[0]= 0;

    cuMemcpyDtoH(copy, memory.blockPrefsums, 64 * sizeof(int));

    cout << "after"<< endl;
    for (int i = 0 ; i < 64; ++i) {
        cout << copy[i] <<" ";
    }
    cout << endl;

    for (int j=0; j <= number_of_local_blocks; ++j) {
//        cout<<"ps " << maxPrefSums[j] << endl;
    }


    for (int j=1; j <= number_of_local_blocks; ++j) {
        maxPrefSums[j] += maxPrefSums[j-1];
    }

    void* args1[] = {&memory.blockPrefsums, &maxPrefSums, &memory.number_of_blocks, &memory.sample_offsets };

    manageResult(cuLaunchKernel(device.prefsumDev1,x_dim, y_dim, 1,THREADS_IN_BLOCK, 1, 1, 0,0, args1, 0), "pref1");
    cuCtxSynchronize();

    cuMemcpyDtoH(copy, memory.blockPrefsums, 2048 * sizeof(int));

    cout << "copy"<< endl;
    for (int j=0; j < 64; ++j) {
        cout <<copy[j] <<" ";
//        cout<<"sof " << memory.sample_offsets[j] << endl;
    }
    cout << endl;
    for (int j=0; j< memory.number_of_blocks; ++j) {
        cout<< memory.sample_offsets[j]<< " ";
    }
    cout << endl;

    cuMemFreeHost(maxPrefSums);
//    exit(1);
}


void sample_rand(Device& device, Memory& memory) {
    fprintf(stderr, "search?\n");
    create_search_tree(memory);
    fprintf(stderr, "counters?\n");
    device.counters(memory);
    fprintf(stderr, "prefsum?\n");
    prefsum(memory, device);
    fprintf(stderr, "scatterr?\n");
    device.scatter(memory);
    fprintf(stderr, "no elo\n");
    memory.moveResult();
    bool all_too_small = true;
    // could be more efficient
    fprintf(stderr, "wherez\n");
    for (int i=0; i< memory.number_of_blocks; ++i){
        int offset = memory.sample_offsets[i];
        fprintf(stderr, "offset %d\n",offset);

        Memory mem(memory, i);
        fprintf(stderr, "herez? %d\n", mem.size);
        //could be more efficient
        if (mem.size > M) {
            all_too_small = false;
            sample_rand(device, mem);
        } else {
            device.odd_even(memory);
        }

    }
    cuMemFreeHost(memory.sample_offsets);
    if (all_too_small) return;
}


void sample_rand(int *to_sort, int size) {
    Device device;
    Memory memory(size);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    cuMemcpyHtoD(memory.deviceToSort, to_sort, size * sizeof(int));

    fprintf(stderr, "before\n");
    sample_rand(device, memory);
    fprintf(stderr, "after\n");

    cuMemcpyDtoH((void *) to_sort, memory.deviceToSort, size * sizeof(int));

    memory.clean();
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(device.cuContext);

}

