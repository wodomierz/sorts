#include "quick_sort_device.h"
#include "../utils/utils.h"

namespace quick {

    Device::Device() {
        manageResult(cuModuleLoad(&cuModule, "quick-sort/quick_sort.ptx"));
        manageResult(cuModuleGetFunction(&gqsortDev, cuModule, "gqsort"));
        manageResult(cuModuleGetFunction(&lqsortDev, cuModule, "lqsort"));
        manageResult(cuModuleGetFunction(&pivotDev, cuModule, "pivot"));
    }


    void Device::gqsort(Block *blocks, int block_count, CUdeviceptr in, CUdeviceptr out, WorkUnit* news) {
        int x_dim = block_count > MAX_GRID_DIM ? MAX_GRID_DIM : block_count;
        int y_dim = ceil_div(block_count, x_dim);
        void *args[]{&blocks, &in, &out, &news};
        safeLaunch1Dim(gqsortDev, x_dim, y_dim, 1 << QUICKTHREADS_POW, args);
    }

    void Device::lqsort(DevArray *seqs, int seqs_count, CUdeviceptr &in, CUdeviceptr &buffer) {
        int x_dim = seqs_count > MAX_GRID_DIM ? MAX_GRID_DIM : seqs_count;
        int y_dim = ceil_div(seqs_count, x_dim);
        void *args[]{&seqs, &in, &buffer};
        safeLaunch1Dim(lqsortDev, x_dim, y_dim, 1 << QUICKTHREADS_POW, args);
    }

    int Device::pivot(CUdeviceptr to_sort, int size) {
        int result;
        int *pivot = cuMemAllocH<int>(1);
        void *args[]{&to_sort, &size, &pivot};
        safeLaunch1Dim(pivotDev, 1, 1, 1,args);
        cuCtxSynchronize();
        result = *pivot;
        cuMemFreeHost(pivot);
        return result;
    }
}
