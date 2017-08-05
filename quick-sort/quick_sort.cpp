
#include <vector>
#include "quick_sort.h"
#include "cuda.h"
#include "../utils/utils.h"
#include "quick_sort_device.h"
#include "quick_shared.h"
#include <list>
#include <algorithm>
#include <cassert>

int pivot(CUdeviceptr to_sort, int size) {
    //TODO
    int *copy = cuMemAllocH<int>(size);
    cuMemcpyDtoH(copy, to_sort, size * sizeof(int));
    PRINT1("co do chuja %d %d %d", copy[0], copy[1], copy[2]);
    std::sort(copy, copy + size);
    return copy[size / 2];
}

int sum_seq_size(std::vector<WorkUnit> work) {
    int result = 0;
    for (WorkUnit unit : work) {
        result += (unit.seq.end - unit.seq.start);
    }
    return result;
}

void checkMem(){
    DevArray *doneArrays1;
    cuMemAllocHost((void**)&doneArrays1, sizeof(DevArray)*10);
    doneArrays1[0] =DevArray(1,2);
    cuMemFreeHost(doneArrays1);
}

void sort(int size, CUdeviceptr &in, CUdeviceptr &out) {
    quick::Device device;

    int start_pivot = pivot(in, size);
    std::vector<WorkUnit> work = {WorkUnit(DevArray(0, size), start_pivot)};
    std::vector<WorkUnit> done;

    int max_seq = 16;

//    print_Devtab(in, size, size,0, "IN");


    while (!work.empty() && work.size() + done.size() < max_seq) {
        int block_size = sum_seq_size(work) / max_seq;
//
        PRINT1("while %d %d %d\n", block_size, sum_seq_size(work), max_seq);
        std::list<Block> block_list;

        int seq_index = -1;
        int total_block_count = 0;
        for (WorkUnit unit : work) {
            //??
            PRINT1("ceil? %d\n", block_size);
            int block_count = ceil_div(unit.seq.end - unit.seq.start, block_size);
            PRINT1("ceil afetr %d\n", block_size);
            total_block_count += block_count;
        }

        //consider vector and register array of this vector??
        Block *blocks = cuMemAllocH<Block>(total_block_count);
        int total_block_index = 0;

        SharedVars* parents = cuMemAllocH<SharedVars>(work.size());
        for (WorkUnit unit : work) {
            PRINT1("ceil? %d\n", block_size);
            int block_count = ceil_div(unit.seq.end - unit.seq.start, block_size);
            PRINT1("ceil afetr %d %d %d %d\n", block_size, block_count, unit.seq.end, unit.seq.start);
            parents[++seq_index] = SharedVars(
                unit.seq,
                unit.seq,
                block_count,
                seq_index
            );
            int i;
            for (i = 0; i < block_count - 1; i++) {
                int bstart = unit.seq.start + block_size * i;
                blocks[total_block_index++] = {
                    {
                        //???
                        {bstart, bstart + block_size},
                        unit.pivot
                    },
                    parents + seq_index
                };
                DevArray &array = blocks[total_block_index - 1].workUnit.seq;
//                PRINT1("add block %d %d %d\n", array.start, array.end, total_block_index -1);
            }
            blocks[total_block_index++] = {
                {
                    {
                        unit.seq.start + block_size * i,
                        unit.seq.end
                    },
                    unit.pivot
                },
                parents +seq_index
            };
            DevArray &array = blocks[total_block_index - 1].workUnit.seq;
//            PRINT1("add block %d %d\n", array.start, array.end);


        }
        PRINT1("alloc ? gqsort %d\n", seq_index);
        //TODO
        WorkUnit *news = cuMemAllocH<WorkUnit>(2*work.size());

        checkMem();
        checkMem();
        PRINT1("before gqsort\n");
//        print_Devtab(out, size, size,0, "BEF GQ");
        cuCtxSynchronize();
        device.gqsort(blocks, total_block_count, in, out, news);
        cuCtxSynchronize();
        checkMem();
        checkMem();
        PRINT1("afetr gqsort %d %d %d\n", blocks[0].sharedVars->seq.start, blocks[1].sharedVars->seq.end, blocks[0].sharedVars->block_count);
//        print_Devtab(in, size, size,0, "IN");
        print_Devtab(out, size, size,0, "GQSORT");
        int* toSort = cuMemAllocH<int>(size);
        cuMemcpyDtoH(toSort, out, sizeof(int) * size);
        std::sort(toSort, toSort + size);
        print_tab(toSort, size, size, "SORTED");
//        PRINT1("%d %d %d %d", news[0].pivot, news[0])
        checkMem();
        //???
        PRINT1("work clear?\n");
        assert(false);
        work.clear();
        checkMem();
        PRINT1("work cleared\n");
        for (int i = 0; i <= seq_index; ++i) {
            WorkUnit &workUnit = news[i];
            checkMem();
            PRINT1("work unit get %d %d %d\n", i, workUnit.seq.start, workUnit.seq.end);
            if (workUnit.seq.end - workUnit.seq.start < size / max_seq) {
                done.push_back(workUnit);
            } else {
                work.push_back(workUnit);
            }
        }
        std::swap(in, out);

        checkMem();
        cuMemFreeHost(blocks);
        PRINT1("free blocks\n");
        cuMemFreeHost(news);
        cuMemFreeHost(parents);
        PRINT1("free news\n");
        checkMem();
    }

    done.insert(done.end(), work.begin(), work.end());
    PRINT1("insert\n");

    //improve adding to done
    checkMem();

    DevArray* doneArrays = cuMemAllocH<DevArray>(done.size());
    PRINT1("alloc done arrays\n");
    for (int i = 0; i < done.size(); i++) {
        PRINT1("bef copied: %d %d %d %d %d\n", i, done[i].seq.start,  done[i].seq.end, done.size(), sizeof(DevArray)*done.size());
        doneArrays[i] = done[i].seq;
        PRINT1("array copied: %d\n", i);
    }
    PRINT1("lq bef\n");
    device.lqsort(doneArrays, done.size(), in, out);
    PRINT1("lq after\n");

    cuMemFreeHost(doneArrays);
    PRINT1("free arrays");
}


void quick_sort_device(CUdeviceptr to_sort, int size) {
    CUdeviceptr in = to_sort;
    CUdeviceptr out = cuAllocInts(size);
    cuMemsetD32(out, 0, size);
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
    cuMemcpyHtoD(in, to_sort, size* sizeof(int));
    quick_sort_device(in, size);
    cuMemcpyDtoH(to_sort, in, sizeof(int) * size);
    cuMemFree(in);

    cuCtxDestroy(cuContext);
}