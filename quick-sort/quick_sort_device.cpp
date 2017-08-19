#include "quick_sort_device.h"
#include "../utils/utils.h"

namespace quick {

    Device::Device() {
        manageResult(cuModuleLoad(&cuModule, "quick-sort/quick_sort.ptx"), "cannot load module");
        manageResult(cuModuleGetFunction(&gqsortDev, cuModule, "gqsort"), "cannot load function");
        manageResult(cuModuleGetFunction(&lqsortDev, cuModule, "lqsort"), "cannot load function");
        manageResult(cuModuleGetFunction(&pivotDev, cuModule, "pivot"), "cannot load function");
    }


    void Device::gqsort(Block *blocks, int block_count, CUdeviceptr in, CUdeviceptr out, WorkUnit *news) {
        int x_dim = block_count > MAX_GRID_DIM ? MAX_GRID_DIM : block_count;
        int y_dim = ceil_div(block_count, x_dim);
        void *args[]{&blocks, &in, &out, &news};
        manageResult(cuLaunchKernel(gqsortDev, x_dim, y_dim, 1, 1 << QUICKTHREADS_POW, 1, 1, 0, 0, args, 0),
                     "running");
        cuCtxSynchronize();
    }

    void Device::lqsort(DevArray *seqs, int seqs_count, CUdeviceptr &in, CUdeviceptr &out) {
        int x_dim = seqs_count > MAX_GRID_DIM ? MAX_GRID_DIM : seqs_count;
        int y_dim = ceil_div(seqs_count, x_dim);
//        PRINT1("launch lqsort %d %d %d\n", seqs_count, x_dim, y_dim);

        void *args[]{&seqs, &in, &out};
        manageResult(cuLaunchKernel(lqsortDev, x_dim, y_dim, 1, 1 << QUICKTHREADS_POW, 1, 1, 0, 0, args, 0),
                     "running");
        cuCtxSynchronize();
    }

    int Device::pivot(CUdeviceptr to_sort, int size) {
        int result;
        int* pivot = cuMemAllocH<int>(1);
        void *args[]{&to_sort, &size, &pivot};
        manageResult(cuLaunchKernel(pivotDev, 1, 1, 1, 1, 1, 1, 0, 0, args, 0),
                     "running");
        cuCtxSynchronize();
        result = *pivot;
        PRINT1("PIVOT %d\n", result);
        cuMemFreeHost(pivot);
        return result;
    }
}
