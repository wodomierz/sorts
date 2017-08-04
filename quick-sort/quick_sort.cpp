//
// Created by Miron Ficak on 03/08/2017.
//

#include <vector>
#include <cudaGL.h>
#include "quick_sort.h"
#include "cuda.h"
#include "../utils/utils.h"
#include "quick_sort_device.h"
#include <list>


int pivot(CUdeviceptr to_sort, int size) {
    //TODO
    return 4;
}

int sum_seq_size(std::vector<WorkUnit> work) {
    int result = 0;
    for (WorkUnit unit : work) {
        result += unit.seq.size;
    }
    return result;
}

void lqsort(int size, CUdeviceptr to_sort, CUdeviceptr out);

void sort(int size, CUdeviceptr &to_sort, CUdeviceptr &out, quick::Device device) {
    int start_pivot = pivot(to_sort, size);
    std::vector<WorkUnit> work = {WorkUnit({0, size}, start_pivot)};
    std::vector<WorkUnit> done;

    int max_seq;
    while (!work.empty() && work.size() + done.size() < max_seq) {
        int block_size = sum_seq_size(work)/max_seq;
//
        std::list<Block> block_list;

        int seq_count = 0;
        int total_block_count=0;
        for (WorkUnit unit : work) {
            //??
            int block_count = ceil_div(unit.seq.size, block_size);
            total_block_count += block_count;
        }

        //consider vector and register array of this vector??
        Block *blocks = cuMemAllocH<Block>(total_block_count);
        int total_block_index = 0;
        for (WorkUnit unit : work) {
            int block_count = ceil_div(unit.seq.size, block_size);

            SharedVars parent = SharedVars(
                unit.seq,
                unit.seq,
                seq_count++,
                block_count
            );
            int i;
            for (i = 0; i < block_count - 1; i++) {
                int bstart = unit.seq.start + block_size * i;
                blocks[total_block_count++] = Block(
                    WorkUnit(
                        //???
                        {bstart, bstart + block_size},
                        unit.pivot
                    ),
                    parent
                );
            }
            blocks[total_block_index++] = Block(
                WorkUnit(
                    DevArray(
                        unit.seq.start + block_size * (block_count - 1),
                        unit.seq.end
                    ),
                    unit.pivot
                ),
                parent
            );
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
    }
    done.insert(done.end(), work.begin(), work.end());

    //improve adding to done
    DevArray* doneArrays = cuMemAllocH<DevArray>(done.size());
    for (int i=0; i< done.size(); i++) {
        doneArrays[i] = done[i].seq;
    }

    device.lqsort(doneArrays, done.size(), to_sort, out);
}


void quick_sort(CUdeviceptr to_sort) {


}