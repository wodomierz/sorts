#include <vector>
#include "quick_sort.h"
#include "cuda.h"
#include "../utils/utils.h"
#include "quick_sort_device.h"
#include "quick_shared.h"
#include <list>
#include <algorithm>
#include <cassert>
int pivot(CUdeviceptr to_sort, int size,quick::Device& device) {
    //TODO
    return device.pivot(to_sort, size);
}

int sum_seq_size(std::vector<WorkUnit> work, int max_seq) {
    int result = 0;
    for (WorkUnit unit : work) {
        result += (unit.seq.end - unit.seq.start)/max_seq;
    }
    return result;
}



void prepareBlocks(Block* blocks, SharedVars* parents, std::vector<WorkUnit>& work, int block_size ){
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
}


void sort(int size, CUdeviceptr &in, CUdeviceptr &out) {
    quick::Device device;

    int start_pivot = pivot(in, size, device);
    std::vector<WorkUnit> work = {WorkUnit(DevArray(0, size), start_pivot)};
    std::vector<WorkUnit> done;

    int block_size = OTHER_SORT_LIM *16;
    int max_seq = ceil_div(size, block_size);



    while (!work.empty() && work.size() + done.size() <= max_seq) {
        int total_block_count = 0;
        for (WorkUnit unit : work) {
            assert(arraySize(unit.seq) > block_size);
            int block_count = ceil_div(unit.seq.end - unit.seq.start, block_size);
            total_block_count += block_count;
        }
        //consider vector and register array of this vector??
        Block *blocks = cuMemAllocH<Block>(total_block_count);


        SharedVars* parents = cuMemAllocH<SharedVars>(work.size());

        prepareBlocks(blocks, parents, work, block_size);

        int seq_num = work.size();
        //TODO
        WorkUnit *news = cuMemAllocH<WorkUnit>(2*seq_num);

        device.gqsort(blocks, total_block_count, in, out, news);
        work.clear();
        for (int i = 0; i < seq_num; ++i) {
            WorkUnit &workUnitL = news[2*i];
            WorkUnit &workUnitR = news[2*i+1];
            if (arraySize(workUnitL.seq) <= block_size) { //diff algo
                    done.push_back(workUnitL);
            } else {
                work.push_back(workUnitL);
            }

            if (arraySize(workUnitR.seq) <= block_size) { //diff algo
                    done.push_back(workUnitR);
            } else {

                work.push_back(workUnitR);
            }
        }
        //todo improve
        cuMemcpyDtoD(in, out, size* sizeof(int));

        cuMemFreeHost(blocks);
        cuMemFreeHost(news);
        cuMemFreeHost(parents);
    }

    done.insert(done.end(), work.begin(), work.end());
    //improve adding to done

    DevArray* doneArrays = cuMemAllocH<DevArray>(done.size());
    for (int i = 0; i < done.size(); i++) {
        doneArrays[i] = done[i].seq;
    }

    //TODO improve
    device.lqsort(doneArrays, done.size(), in, out);
    cuMemFreeHost(doneArrays);
}


void quick_sort_device(CUdeviceptr to_sort, int size) {
    CUdeviceptr in = to_sort;
    CUdeviceptr buffer = cuAllocInts(size);
    cuMemsetD32(buffer, 0, size);
    sort(size, in, buffer);
    cuMemFree(buffer);
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