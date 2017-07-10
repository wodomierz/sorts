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
//#include <thrust/scan.h>
//#include <thrust/execution_policy.h>
//
#define THREADS_IN_BLOCK 1024

using namespace std;

int *create_search_tree(int *to_sort) {
    int *sample = new int[S_SIZE];
    int *tree = new int[S_SIZE];
    std::copy(to_sort, to_sort + S_SIZE, sample);
    std::sort(sample, sample + S_SIZE);

    int iteratr = 0;
    for (int i = 2; i <= S_SIZE; i *= 2) {
        for (int j = 1; j < i; j += 2) {
            tree[iteratr++] = sample[j * (S_SIZE / i)];
        }
    }
    delete[] sample;
    return tree;
}

inline void prefsum(CUdeviceptr blockPrefsums, int size, CUmodule cuModule, CUfunction prefsumDev,
CUfunction prefsumDev1
) {
    int* maxPrefSums;
    //here can be bug if size is big
    int number_of_local_blocks = (size/2 + THREADS_IN_BLOCK -1)/ THREADS_IN_BLOCK;

    int max_grid_dim_x = 32768;
    int x_dim = number_of_local_blocks > max_grid_dim_x ? max_grid_dim_x : number_of_local_blocks;
    int y_dim = (number_of_local_blocks + x_dim - 1) / x_dim;
    cout << "pref dziwne "<< x_dim << " " << y_dim<< endl;
    cuMemAllocHost((void**)&maxPrefSums, (number_of_local_blocks + 1) * sizeof(int));
//    CUdeviceptr maxPrevSumDev;
//    cuMemAlloc(&maxPrevSumDev, (number_of_local_blocks + 1) * sizeof(int));

//memcpy ?

//    CUfunction prefsumDev;
//    manageResult(cuModuleGetFunction(&prefsumDev, cuModule, "prefsum"), "cannot load function");
//
//    CUfunction prefsumDev1;
//    manageResult(cuModuleGetFunction(&prefsumDev1, cuModule, "prefsum1"), "cannot load function");


    void* args[] = {&blockPrefsums, &maxPrefSums };
    std::cout << "running"<< std::endl;
    manageResult(cuLaunchKernel(prefsumDev,x_dim, y_dim, 1,THREADS_IN_BLOCK, 1, 1, 0,0, args, 0), "pref2");
    cuCtxSynchronize();

//    cuMemcpyDtoH((void*)maxPrefSums, maxPrevSumDev, (number_of_local_blocks + 1) * sizeof(int));
    maxPrefSums[0]= 0;

    for (int j=1; j <= number_of_local_blocks; ++j) {
        maxPrefSums[j] += maxPrefSums[j-1];
    }
    manageResult(cuLaunchKernel(prefsumDev1,x_dim, y_dim, 1,THREADS_IN_BLOCK, 1, 1, 0,0, args, 0), "pref1");
    cuCtxSynchronize();

//    cuMemFree(maxPrevSumDev);
    cuMemFreeHost(maxPrefSums);


}
void sample_rand(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
    CUmodule cuModule = (CUmodule) 0;
    manageResult(cuModuleLoad(&cuModule, "sample-rand/sample_rand.ptx"), "cannot load module");
//   CUfunction bitonic_merge;
//    manageResult(cuModuleGetFunction(&bitonic_merge, cuModule, "bitonic_merge"), "cannot load function");
//    CUfunction bitonic_triangle_merge;
//    manageResult(cuModuleGetFunction(&bitonic_triangle_merge, cuModule, "bitonic_triangle_merge"),
//                 "cannot load function");
    int c = 3;
    int n;
    int power_n;
    //fit n to power of 2
    for (n = 1, power_n = 0; n < size; n <<= 1, power_n++);
//    int half_size = n / 2;
    int numberOfBlocks = (n + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int max_grid_dim_x = 32768;
    int x_dim = numberOfBlocks > max_grid_dim_x ? max_grid_dim_x : numberOfBlocks;
    int y_dim = (numberOfBlocks + x_dim - 1) / x_dim;
    cout << x_dim << " " << y_dim << endl;
//    int z_dim = 1;
//    if (y_dim > max_grid_dim_x) {
//        z_dim = (y_dim + max_grid_dim_x - 1) / max_grid_dim_x;
//        y_dim = max_grid_dim_x;
//    }


    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, size * sizeof(int));

    CUdeviceptr out;
    cuMemAlloc(&out, size * sizeof(int));


    cuMemcpyHtoD(deviceToSort, to_sort, size * sizeof(int));

//    int s_size = 1024;
//    int s_pow = 10;

    int *s_tree = create_search_tree(to_sort);
    cuMemHostRegister((void *) s_tree, S_SIZE* sizeof(int), 0);

    CUdeviceptr bstPtr;
    cuMemAlloc(&bstPtr, S_SIZE * sizeof(int));
    cuMemcpyHtoD(bstPtr, s_tree, S_SIZE * sizeof(int));

    CUdeviceptr blockPrefsums;
    cuMemAlloc(&blockPrefsums, S_SIZE * numberOfBlocks * sizeof(int));


    CUfunction prefsumDev;
    manageResult(cuModuleGetFunction(&prefsumDev, cuModule, "prefsum"), "cannot load function");

    CUfunction prefsumDev1;
    manageResult(cuModuleGetFunction(&prefsumDev1, cuModule, "prefsum1"), "cannot load function");

    CUfunction counters;
    manageResult(cuModuleGetFunction(&counters, cuModule, "counters"), "cannot load function");

    void *args1[] = {&deviceToSort, &bstPtr, &blockPrefsums, &numberOfBlocks};
    manageResult(cuLaunchKernel(counters, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0), "running");
    cuCtxSynchronize();


    prefsum(blockPrefsums, S_SIZE * numberOfBlocks, cuModule, prefsumDev, prefsumDev1);
    std::cout << "after prefsum" << std::endl;

    CUfunction scatter;
    manageResult(cuModuleGetFunction(&scatter, cuModule, "scatter"), "cannot load function");

    void *args2[] {&deviceToSort, &out, &bstPtr, &blockPrefsums, &numberOfBlocks};
    manageResult(cuLaunchKernel(scatter, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0), "running");
    cuCtxSynchronize();

    cuMemcpyDtoH((void *) to_sort, out, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemFree(out);
    cuMemFree(blockPrefsums);
    cuMemFree(bstPtr);

    cuMemHostUnregister(s_tree);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);

    delete[] s_tree;
//    return result;


}
