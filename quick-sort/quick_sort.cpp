//
// Created by Miron Ficak on 03/08/2017.
//

#include <vector>
#include "quick_sort.h"
#include "cuda.h"
#include "../utils/utils.h"
#include "quick_sort_device.h"
#include "quick_shared.h"
#include <list>
#include <algorithm>

int pivot(CUdeviceptr to_sort, int size) {
    //TODO
    int *copy = cuMemAllocH<int>(size);
    cuMemcpyDtoH(copy, to_sort, size * sizeof(int));
    std::sort(copy, copy + size);
    return copy[size / 2];
}

int sum_seq_size(std::vector<WorkUnit> work) {
    int result = 0;
    for (WorkUnit unit : work) {
        result += unit.seq.size;
    }
    return result;
}

void sort(int size, CUdeviceptr &to_sort, CUdeviceptr &out) {
    quick::Device device;

    int start_pivot = pivot(to_sort, size);
    std::vector<WorkUnit> work = {WorkUnit(DevArray(0, size), start_pivot)};
    std::vector<WorkUnit> done;

    int max_seq = 1024;
    PRINT1("bef while\n");
    while (!work.empty() && work.size() + done.size() < max_seq) {
        int block_size = sum_seq_size(work) / max_seq;
//
        PRINT1("while %d\n", block_size);
        std::list<Block> block_list;

        int seq_count = 0;
        int total_block_count = 0;
        for (WorkUnit unit : work) {
            //??
            PRINT1("ceil? %d\n", block_size);
            int block_count = ceil_div(unit.seq.size, block_size);
            PRINT1("ceil afetr %d\n", block_size);
            total_block_count += block_count;
        }

        //consider vector and register array of this vector??
        Block *blocks = cuMemAllocH<Block>(total_block_count);
        int total_block_index = 0;
        for (WorkUnit unit : work) {
            PRINT1("ceil? %d\n", block_size);
            int block_count = ceil_div(unit.seq.size, block_size);
            PRINT1("ceil afetr %d\n", block_size);
            SharedVars parent = SharedVars(
                unit.seq,
                unit.seq,
                seq_count++,
                block_count
            );
            int i;
            for (i = 0; i < block_count - 1; i++) {
                int bstart = unit.seq.start + block_size * i;
                blocks[total_block_count++] = {
                {
                    //???
                    { bstart, bstart + block_size },
                    unit.pivot
                },
                parent
                };
                blocks[total_block_index++] = {
                    {
                        {
                            unit.seq.start + block_size * (block_count - 1),
                            unit.seq.end
                        },
                        unit.pivot
                    },
                    parent
                };
            }
        }
        //TODO
        WorkUnit *news = cuMemAllocH<WorkUnit>(seq_count);

        device.gqsort(blocks, total_block_count, to_sort, out, news);

        //???
        work.clear();
        for (int i = 0; i < seq_count; ++i) {
            WorkUnit &workUnit = news[i];
            if (workUnit.seq.size < size / max_seq) {
                done.push_back(workUnit);
            } else {
                work.push_back(workUnit);
            }
        }
        std::swap(to_sort, out);


        cuMemFreeHost(blocks);
        cuMemFreeHost(news);
    }
    done.insert(done.end(), work.begin(), work.end());

    //improve adding to done
    DevArray *doneArrays = cuMemAllocH<DevArray>(done.size());
    for (int i = 0; i < done.size(); i++) {
        doneArrays[i] = done[i].seq;
    }

    device.lqsort(doneArrays, done.size(), to_sort, out);
    cuMemFreeHost(doneArrays);
}


void quick_sort_device(CUdeviceptr to_sort, int size) {
    CUdeviceptr in = to_sort;
    CUdeviceptr out = cuAllocD<int>(size);
    sort(size, in, out);

    if (out != to_sort) {
        cuMemcpyDtoD(to_sort, out, sizeof(int) * size);
        cuMemFree(out);
    } else {
        cuMemFree(in);
    }
}

void quick_sort(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    CUcontext cuContext;
    manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");

    PRINT1("tutej\n");
    cuMemHostRegister(to_sort, sizeof(int) * size, 0);
    CUdeviceptr in = cuAllocD<int>(size);
    quick_sort_device(in, size);
    cuMemcpyDtoH(to_sort, in, sizeof(int) * size);
    cuMemFree(in);

    cuCtxDestroy(cuContext);
}