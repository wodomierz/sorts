//
// Created by Miron Ficak on 03/08/2017.
//

#include "quick_sort_device.h"
#include "../utils/utils.h"

namespace quick {

    Device::Device() {
        manageResult(cuModuleLoad(&cuModule, "quick-sort/quick_sort.ptx"), "cannot load module");
        manageResult(cuModuleGetFunction(&gqsortDev, cuModule, "gqsort"), "cannot load function");
        manageResult(cuModuleGetFunction(&lqsortDev, cuModule, "lqsort"), "cannot load function");
    }

    void Device::gqsort(Block *blocks, int block_count, CUdeviceptr in, CUdeviceptr out, WorkUnit *news) {
        int x_dim = block_count > MAX_GRID_DIM? MAX_GRID_DIM : block_count;
        int y_dim = ceil_div(block_count, x_dim);
        void* args[]{&blocks, &in, &out, &news};
        manageResult(cuLaunchKernel(gqsortDev, x_dim, y_dim,1, QUICK_THREADS_IN_BLOCK, 1, 1, 0,0,0,0), "running");
    }

    void Device::lqsort(DevArray *seqs, int seqs_count, CUdeviceptr& in, CUdeviceptr& out) {
        int x_dim = seqs_count > MAX_GRID_DIM? MAX_GRID_DIM : seqs_count;
        int y_dim = ceil_div(seqs_count, x_dim);
        CUdeviceptr* ptr_in = cuMemAllocH<CUdeviceptr>(1);
        CUdeviceptr* ptr_out = cuMemAllocH<CUdeviceptr>(1);
        *ptr_in = in;
        *ptr_out = out;


        void* args[]{&seqs, &ptr_in, &ptr_out};
        manageResult(cuLaunchKernel(gqsortDev, x_dim, y_dim,1, QUICK_THREADS_IN_BLOCK, 1, 1, 0,0,0,0), "running");
        in = *ptr_in;
        out = *ptr_out;
    }
}
