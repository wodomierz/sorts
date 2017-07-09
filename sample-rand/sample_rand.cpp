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



int* create_search_tree(int* to_sort, int sample_size) {
    int* sample = new int[sample_size];
    int* tree = new int[sample_size];
    std::copy(to_sort, to_sort + sample_size, sample);
    std::sort(sample, sample + sample_size);

    int iteratr = 0;
    for (int i = 2; i <= sample_size; i *= 2) {
        for (int j=1; j < i; j+=2) {
            tree[iteratr++] = sample[j * (sample_size / i)];
        }
    }
    delete[] sample;
    return tree;
}



void sample_rand(int * to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
    CUmodule cuModule = (CUmodule)0;
    manageResult(cuModuleLoad(&cuModule, "bitonic/bitonic_sort.ptx"), "cannot load module");
    CUfunction bitonic_merge;
    manageResult(cuModuleGetFunction(&bitonic_merge, cuModule, "bitonic_merge" ) , "cannot load function");
    CUfunction bitonic_triangle_merge;
    manageResult(cuModuleGetFunction(&bitonic_triangle_merge, cuModule,"bitonic_triangle_merge"), "cannot load function");

    int n;
    int power_n;
    //fit n to power of 2
    for (n = 1, power_n=0; n < size; n <<= 1, power_n++);
    int half_size = n/2;
    int numberOfBlocks = (half_size + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int max_grid_dim_x = 32768;
    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;
    int z_dim = 1;
    if (y_dim > max_grid_dim_x) {
        z_dim = (y_dim + max_grid_dim_x -1) / max_grid_dim_x;
        y_dim = max_grid_dim_x;
    }

    cuMemHostRegister((void*) to_sort, size * sizeof(int), 0);
    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));
    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));

    int s_size = 10;


    int* s_tree = create_search_tree(to_sort, s_size);




    cuMemcpyDtoH((void*)to_sort, deviceToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
//    return result;


}
