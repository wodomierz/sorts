#include <vector>
#include "quick_sort.h"
#include "quick_sort_device.h"
#include <list>
#include <ctime>

int pivot(CUdeviceptr to_sort, int size, quick::Device &device) {
    return device.pivot(to_sort, size);
}

inline Block block(int seq_index, SharedVars *parents, WorkUnit &unit, int &start, int end) {
    return {
        {
            {start, end},
            unit.pivot
        },
        parents + seq_index
    };
};

inline Block middleBlock(int seq_index, SharedVars *parents, int block_size, WorkUnit &unit, int bstart) {
    return block(seq_index, parents, unit, bstart, bstart + block_size);
};

inline Block lastBlock(int seq_index, SharedVars *parents, WorkUnit &unit, int bstart) {
    return block(seq_index, parents, unit, bstart, unit.seq.end);
};


void prepareBlocks(Block *blocks, SharedVars *parents, std::vector<WorkUnit> &work, int block_size) {
    int total_block_index = 0;
    int seq_index = -1;

    for (WorkUnit unit : work) {
        int block_count = ceil_div(unit.seq.end - unit.seq.start, block_size);
        parents[++seq_index] = SharedVars(
            unit.seq,
            unit.seq,
            block_count,
            seq_index
        );
        int i;
        for (i = 0; i < block_count - 1; i++) {
            blocks[total_block_index++] = middleBlock(seq_index, parents, block_size, unit,
                                                      unit.seq.start + block_size * i);
        }
        blocks[total_block_index++] = lastBlock(seq_index, parents, unit, unit.seq.start + block_size * i);
    }
}


void sort(int size, CUdeviceptr &in, CUdeviceptr &out) {
    quick::Device device;

    int start_pivot = pivot(in, size, device);
    std::vector<WorkUnit> work = {WorkUnit(DevArray(0, size), start_pivot)};
    std::vector<WorkUnit> done;

    int max_seq = ceil_div(size, QUICK_BLOCK_SIZE);

    Block *blocks = cuMemAllocH<Block>(2 * max_seq);
    SharedVars *parents = cuMemAllocH<SharedVars>(max_seq);
    WorkUnit *news = cuMemAllocH<WorkUnit>(2 * max_seq);


    while (!work.empty() && work.size() + done.size() <= max_seq) {
        int total_block_count = 0;
        for (WorkUnit unit : work) {
            int block_count = ceil_div(unit.seq.end - unit.seq.start, QUICK_BLOCK_SIZE);
            total_block_count += block_count;
        }
        int seq_num = work.size();
        prepareBlocks(blocks, parents, work, QUICK_BLOCK_SIZE);
        device.gqsort(blocks, total_block_count, in, out, news);

        cuCtxSynchronize();
        work.clear();
        for (int i = 0; i < seq_num; ++i) {
            for (int j = 0; j < 2; ++j) {
                WorkUnit &workUnit = news[2 * i + j];
                if (arraySize(workUnit.seq) < QUICK_BLOCK_SIZE) { //diff algo
                    if (arraySize(workUnit.seq) > 1) {
                        done.push_back(workUnit);
                    }
                } else {
                    work.push_back(workUnit);
                }
            }
        }
        cuMemcpyDtoD(in, out, size * sizeof(int));
    }
    done.insert(done.end(), work.begin(), work.end());

    DevArray *doneArrays = cuMemAllocH<DevArray>(done.size());
    for (int i = 0; i < done.size(); i++) {
        doneArrays[i] = done[i].seq;
    }

    if (done.size() > 0) {
        device.lqsort(doneArrays, done.size(), in, out);
    }

    cuMemFreeHost(doneArrays);
    cuMemFreeHost(blocks);
    cuMemFreeHost(news);
    cuMemFreeHost(parents);
}


void quick_sort_device(CUdeviceptr to_sort, int size) {
    CUdeviceptr in = to_sort;
    CUdeviceptr buffer = cuAllocInts(size);
    sort(size, in, buffer);
    cuMemFree(buffer);
}

void quick_sort(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    CUcontext cuContext;
    manageResult(cuDeviceGet(&cuDevice, 0));
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice));

    cuMemHostRegister(to_sort, sizeof(int) * size, 0);
    CUdeviceptr in = cuAllocD<int>(size);
    cuMemcpyHtoD(in, to_sort, size * sizeof(int));

    quick_sort_device(in, size);

    cuMemcpyDtoH(to_sort, in, sizeof(int) * size);
    cuMemHostUnregister(to_sort);
    cuMemFree(in);

    cuCtxDestroy(cuContext);
}