//
// Created by Miron Ficak on 03/08/2017.
//

#include <vector>
#include "quick_sort.h"
#include "cuda.h"
#include "../utils/utils.h"
#include "quick_sort_device.h"






struct SharedVars {
    DevArray seq1;
    DevArray seq2;
    int block_count;

    SharedVars(const DevArray &seq1, const DevArray &seq2, int block_count) :
            seq1(seq1), seq2(seq2), block_count(block_count) {}

};

struct Block {
    WorkUnit workUnit;
    SharedVars sharedVars;

    Block(const WorkUnit &workUnit, const SharedVars &sharedVars) :
            workUnit(workUnit), sharedVars(sharedVars) {}
};

int pivot(CUdeviceptr to_sort, int size) {
    //TODO
    return 4;
}

int sum_seq_size(std::vector<WorkUnit> work) {
    int result = 0;
    for (WorkUnit unit : work) {
        result += unit.d.size;
    }
}

void lqsort();

void sort(int size, CUdeviceptr to_sort, CUdeviceptr out, quick::Device device) {
    int start_pivot = pivot(to_sort, size);
    std::vector<WorkUnit> work = {WorkUnit(DevArray(to_sort, 0, size), start_pivot)};
    std::vector<WorkUnit> done;

    while (!work.empty() && work.size() + done.size() < MAX_SEQ) {
        int block_size = sum_seq_size(work);
        std::vector<Block> blocks;

        for (WorkUnit unit : work) {

            int block_count = ceil_div(unit.d.size, block_size);

            SharedVars parent = SharedVars(
                    unit.d,
                    unit.d,
                    block_count
            );
            for (int i = 0; i < block_size - 1; i++) {
                int bstart = unit.d.start + block_size * i;
                blocks.push_back(
                        Block(
                                WorkUnit(
                                        DevArray(
                                                unit.d.d,
                                                bstart,
                                                bstart + block_size
                                        ),
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
                                            unit.d.d,
                                            unit.d.start + block_size*(block_count-1),
                                            unit.d.end
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
                if (workUnit.d.size < size / MAX_SEQ) {
                    done.push_back(workUnit);
                } else {
                    work.push_back(workUnit);
                }
            }
        }
        done.insert(done.end(), work.begin(), work.end());
        lqsort();

    }


}


void quick_sort(CUdeviceptr to_sort) {


}