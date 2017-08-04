//
// Created by Miron Ficak on 03/08/2017.
//

#include <vector>
#include <cudaGL.h>
#include "quick_sort.h"
#include "cuda.h"
#include "../utils/utils.h"
#include "quick_sort_device.h"


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

void sort(int size, CUdeviceptr to_sort, CUdeviceptr out, quick::Device device) {
    int start_pivot = pivot(to_sort, size);
    std::vector<WorkUnit> work = {WorkUnit({0, size}, start_pivot)};
    std::vector<WorkUnit> done;

    while (!work.empty() && work.size() + done.size() < MAX_SEQ) {
        int block_size = sum_seq_size(work);
        std::vector<Block> blocks;

        for (WorkUnit unit : work) {

            int block_count = ceil_div(unit.seq.size, block_size);

            SharedVars parent = SharedVars(
                unit.seq,
                unit.seq,
                block_count,
                block_count
            );
            for (int i = 0; i < block_size - 1; i++) {
                int bstart = unit.seq.start + block_size * i;
                blocks.push_back(
                    Block(
                        WorkUnit(
                            //???
                            {bstart, bstart + block_size},
                            unit.pivot
                        ),
                        parent
                    )
                );
            }
            blocks.push_back(
                Block(
                    WorkUnit(
                        DevArray(
                            unit.seq.start + block_size * (block_count - 1),
                            unit.seq.end
                        ),
                        unit.pivot
                    ),
                    parent
                )
            );
            //TODO
            std::vector<WorkUnit> news = device.gqsort();

            //???
            work = std::vector<WorkUnit>();
            for (WorkUnit workUnit: news) {
                if (workUnit.seq.size < size / MAX_SEQ) {
                    done.push_back(workUnit);
                } else {
                    work.push_back(workUnit);
                }
            }
        }
        done.insert(done.end(), work.begin(), work.end());
        lqsort(size, to_sort, out);

        std::swap(to_sort, out);
    }


}


void quick_sort(CUdeviceptr to_sort) {


}