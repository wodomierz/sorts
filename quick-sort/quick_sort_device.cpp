
#include "quick_sort_device.h"
#include "../utils/utils.h"

namespace quick {

    Device::Device() {
        manageResult(cuModuleLoad(&cuModule, "quick-sort/quick_sort.ptx"), "cannot load module");
        manageResult(cuModuleGetFunction(&gqsortDev, cuModule, "gqsort"), "cannot load function");
        manageResult(cuModuleGetFunction(&lqsortDev, cuModule, "lqsort"), "cannot load function");
    }

    void Device::gqsort(Block *blocks, int block_count, CUdeviceptr in, CUdeviceptr out, WorkUnit *news) {
        int x_dim = block_count > MAX_GRID_DIM ? MAX_GRID_DIM : block_count;
        int y_dim = ceil_div(block_count, x_dim);
        void *args[]{&blocks, &in, &out, &news};
//        PRINT1("launch gqsort %d %d %d %d\n", block_count, QUICK_THREADS_IN_BLOCK, x_dim, y_dim);
//        PRINT1("launch gqsort %d %d %d %d %d %d\n", blocks[0].workUnit.seq.start, blocks[0].workUnit.seq.end,
//               blocks[0].sharedVars->seq.start,
//               blocks[0].sharedVars->seq.end,
//               blocks[0].workUnit.pivot,
//               blocks[0].sharedVars->block_count
//        );
        manageResult(cuLaunchKernel(gqsortDev, x_dim, y_dim, 1, QUICK_THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0),
                     "running");
        cuCtxSynchronize();
    }

    void Device::lqsort(DevArray *seqs, int seqs_count, CUdeviceptr &in, CUdeviceptr &out) {
        int x_dim = seqs_count > MAX_GRID_DIM ? MAX_GRID_DIM : seqs_count;
        int y_dim = ceil_div(seqs_count, x_dim);
//        CUdeviceptr *ptr_in = cuMemAllocH<CUdeviceptr>(1);
//        CUdeviceptr *ptr_out = cuMemAllocH<CUdeviceptr>(1);
//        *ptr_in = in;
//        *ptr_out = out;

        int *debugger = cuMemAllocH<int>(100000);
        for (int i = 0; i < 100000; ++i) debugger[i] = -2;

        PRINT1("launch lqsort %d %d %d %d\n", seqs_count, QUICK_THREADS_IN_BLOCK, x_dim, y_dim);

        void *args[]{&seqs, &in, &out, &debugger};
        manageResult(cuLaunchKernel(lqsortDev, x_dim, y_dim, 1, QUICK_THREADS_IN_BLOCK, 1, 1, 0, 0, args, 0),
                     "running");
        cuCtxSynchronize();
        for (int i = 0; i < 50; ++i) {
            PRINT1("deb %d %d\n", i, debugger[i]);
        }

//        in = *ptr_in;
//        out = *ptr_out;
//        cuMemFreeHost(ptr_in);
//        cuMemFreeHost(ptr_out);
    }
}
