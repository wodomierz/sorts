#include <cstdio>
#include <iostream>
#include "../utils/utils.h"

using namespace std;
static int THREADS_IN_BLOCK = 1024;

void radixsort(int* to_sort, int size){
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0), "device");

//    int MAX_INT = numeric_limits<int>::max();

    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "ctx");
    CUmodule cuModule = (CUmodule)0;
    manageResult(cuModuleLoad(&cuModule, "radix/radixsort.ptx"), "module");
    CUfunction sort;
    manageResult(cuModuleGetFunction(&sort, cuModule, "sort"), "func");
    CUfunction prefixSum;
    manageResult(cuModuleGetFunction(&prefixSum, cuModule, "prefixSum"), "func");



//    int numberOfBlocks = (size+2048-1)/2048;
    int numberOfBlocks = (size+2048-1)/2048;

    // int n = 2048*numberOfBlocks;
    int* prefixSums;
    CUdeviceptr localSums;
    CUdeviceptr tab[2];

    int* biggerTab = to_sort;

    int n=size;
    cuMemHostRegister((void*) biggerTab, sizeof(int)*n, 0);
    cuMemAllocHost((void **) &prefixSums, (numberOfBlocks + 1)*sizeof(int));

    cuMemAlloc(&tab[0], n * sizeof(int));
    cuMemAlloc(&tab[1], n * sizeof(int));
    cuMemAlloc(&localSums, n * sizeof(int));

    cuMemcpyHtoD(tab[0], biggerTab, n * sizeof(int));

    prefixSums[0] = 0;

    int number_of_zeros = 0;
    int mask = 0;
    void* args[5] =  { NULL,&localSums, &prefixSums, &n, &mask};
    void* args1[7] = { NULL, NULL, &localSums, &prefixSums,&mask, &n, &number_of_zeros};


    for (mask=0; mask < 31 ; mask++ ) {
        args[0] = &tab[mask%2];

        cuLaunchKernel(prefixSum, numberOfBlocks, 1, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0);

        cuCtxSynchronize();

        for (int j=1; j <= numberOfBlocks; ++j) {
            prefixSums[j] += prefixSums[j-1];
        }
        number_of_zeros = n - prefixSums[numberOfBlocks];

        args1[0] = &tab[mask%2];
        args1[1] = &tab[1 - mask%2];

        cuLaunchKernel(sort, numberOfBlocks, 1, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0);
        cuCtxSynchronize();
    }

    cuMemcpyDtoH((void*) to_sort, tab[1], size * sizeof(int));

    cuMemFreeHost((void*)prefixSums);
    cuMemFree(tab[0]);
    cuMemFree(tab[1]);
    cuMemFree(localSums);
    cuCtxDestroy(cuContext);
}
