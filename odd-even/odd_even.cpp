#include "cuda.h"
#include <cstdio>
#include <iostream>
#include "../utils/utils.h"
#include "../utils/BaseKernelData.h"
#include "odd_even.h"

using namespace std;

void odd_even(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    manageResult(cuDeviceGet(&cuDevice, 0));
    CUcontext cuContext;
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice));
    CUmodule cuModule = (CUmodule) 0;
    manageResult(cuModuleLoad(&cuModule, "odd-even/odd_even.ptx"));
    CUfunction odd_even_phase1;
    manageResult(cuModuleGetFunction(&odd_even_phase1, cuModule, "odd_even_phase1"));
    CUfunction odd_even_phase2;
    manageResult(cuModuleGetFunction(&odd_even_phase2, cuModule, "odd_even_phase2"));
    CUfunction odd_even;
    manageResult(cuModuleGetFunction(&odd_even, cuModule, "odd_even"));

    int algined_size = expand_to_power_of_2(size, ODD_EVEN_BLOCK_POW);
    int delta = algined_size - size;

    int n;
    int power_n;
    for (n = 1, power_n = 0; n < algined_size; n <<= 1, power_n++);

    BaseKernelData baseData1(algined_size, ODD_EVEN_BLOCK);
    BaseKernelData baseData2(n / 2, ODD_EVEN_THREADS);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);

    CUdeviceptr deviceToSort;
    cuMemAlloc(&deviceToSort, algined_size * sizeof(int));
    CUdeviceptr shelfedToSort = addIntOffset(deviceToSort, delta);
    cuMemcpyHtoD(shelfedToSort, to_sort, size * sizeof(int));
    cuMemsetD32(deviceToSort, 0, delta);

    void *args[] = {&deviceToSort};
    safeLaunch1Dim(odd_even, baseData1.x_dim, baseData1.y_dim, ODD_EVEN_THREADS, args);
    for (int pow__half_batch = ODD_EVEN_BLOCK_POW; pow__half_batch <= power_n - 1; pow__half_batch++) {
        int half_batch = 1 << pow__half_batch;
        void *args1[3] = {&deviceToSort, &pow__half_batch, &algined_size};
        safeLaunch1Dim(odd_even_phase1, baseData2.x_dim, baseData2.y_dim, ODD_EVEN_THREADS, args1);
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {
            void *args2[4] = {&deviceToSort, &d_power, &half_batch, &algined_size};
            safeLaunch1Dim(odd_even_phase2, baseData2.x_dim, baseData2.y_dim, ODD_EVEN_THREADS, args2);
        }

    }

    cuMemcpyDtoH((void *) to_sort, shelfedToSort, size * sizeof(int));

    cuMemFree(deviceToSort);
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(cuContext);
}

