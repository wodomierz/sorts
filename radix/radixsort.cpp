#include <cstdio>
#include <iostream>
#include "../utils/utils.h"
#include "radixsort.h"

using namespace std;
static int ThreadsInBlock = RADIX_THREADS;
static int BlockSize = ThreadsInBlock * RADIX_ELEMENTS;

//size must be bigger than BlockSize !

void radixsort(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0));

    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice));
    CUmodule cuModule = (CUmodule) 0;
    manageResult(cuModuleLoad(&cuModule, "radix/radixsort.ptx"));
    CUfunction sort;
    manageResult(cuModuleGetFunction(&sort, cuModule, "sort"));
    CUfunction count_and_prefsum;
    manageResult(cuModuleGetFunction(&count_and_prefsum, cuModule, "count_and_pref_sum"));
    CUfunction pref_sum;
    manageResult(cuModuleGetFunction(&pref_sum, cuModule, "pref_sum"));
    CUfunction add;
    manageResult(cuModuleGetFunction(&add, cuModule, "add"));
    CUfunction one_block_prefsum;
    manageResult(cuModuleGetFunction(&one_block_prefsum, cuModule, "one_block_prefsum"));

    int numberOfBlocks = ceil_div(size, BlockSize);

    CUdeviceptr localSums;
    CUdeviceptr tab[2];

    int *biggerTab = to_sort;

    int n = size;
    cuMemHostRegister((void *) biggerTab, sizeof(int) * n, 0);

    cuMemAlloc(&tab[0], n * sizeof(int));
    cuMemAlloc(&tab[1], n * sizeof(int));
    cuMemAlloc(&localSums, n * sizeof(int));


    cuMemcpyHtoD(tab[0], biggerTab, n * sizeof(int));

    int mask = 0;
    CUdeviceptr number_of_zeros_dev = cuAllocD<int>(1);

    vector<CudaArray> prefsum_arrays;
    int prefsum_arrays_size = n;
    while (prefsum_arrays_size > BlockSize) {
        prefsum_arrays_size = ceil_div(prefsum_arrays_size, BlockSize);
        prefsum_arrays.push_back({cuAllocD<int>(prefsum_arrays_size + 1), prefsum_arrays_size});
        cuMemsetD32(prefsum_arrays.back().array, 0, 1);
    }
    void *args[5] = {NULL, &localSums, &(*prefsum_arrays.begin()), &n, &mask};

    void *args1[7] = {NULL, NULL, &localSums, &(*prefsum_arrays.begin()), &mask, &n, &numberOfBlocks};

    for (mask = 0; mask < 31; mask++) {
        args[0] = &tab[mask % 2];
        cuLaunchKernel(count_and_prefsum, numberOfBlocks, 1, 1, ThreadsInBlock, 1, 1, 0, 0, args, 0);

        for (int i = 1; i < prefsum_arrays.size(); ++i) {
            CUdeviceptr shiftedArray = addIntOffset(prefsum_arrays[i - 1].array, 1);
            void *args[] = {&shiftedArray, &prefsum_arrays[i].array, &prefsum_arrays[i - 1].size};

            cuLaunchKernel(pref_sum, prefsum_arrays[i].size, 1, 1, ThreadsInBlock, 1, 1, 0, 0, args, 0);

        }
        CUdeviceptr shiftedArray = addIntOffset(prefsum_arrays.back().array, 1);
        void *args_one[] = {&shiftedArray, &prefsum_arrays.back().size};

        cuLaunchKernel(one_block_prefsum, 1, 1, 1, ThreadsInBlock, 1, 1, 0, 0, args_one, 0);
        for (int i = prefsum_arrays.size() - 1; i > 0; i--) {
            CUdeviceptr shiftedArray = addIntOffset(prefsum_arrays[i - 1].array, 1);
            void *args[] = {&shiftedArray, &prefsum_arrays[i].array, &prefsum_arrays[i - 1].size};
            cuLaunchKernel(add, prefsum_arrays[i].size, 1, 1, ThreadsInBlock, 1, 1, 0, 0, args, 0);
        }

        args1[0] = &tab[mask % 2];
        args1[1] = &tab[1 - mask % 2];
        cuLaunchKernel(sort, numberOfBlocks, 1, 1, ThreadsInBlock, 1, 1, 0, 0, args1, 0);
    }

    cuMemcpyDtoH((void *) to_sort, tab[1], size * sizeof(int));

    cuMemHostUnregister(to_sort);

    for (auto ptr: prefsum_arrays) {
        cuMemFree(ptr.array);
    }
    cuMemFree(number_of_zeros_dev);
    cuMemFree(tab[0]);
    cuMemFree(tab[1]);
    cuMemFree(localSums);
    cuCtxDestroy(cuContext);
}
