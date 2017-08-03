//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_QUICK_SORT_H
#define SORTS_QUICK_SORT_H

#include <cuda.h>

#define QUICK_THREADS_IN_BLOCK 1024
#define QUICK_BLOCK_SIZE QUICK_THREADS_IN_BLOCK
#define OTHER_SORT_LIM 2048
#define MAX_SEQ 1024 //????

namespace quick {
  static const int THREADS_NUM = QUICK_THREADS_IN_BLOCK
}

struct SharedVars {
    const int seq_index;
    DevArray seq1;
    DevArray seq2;
    int block_count;

    SharedVars(const DevArray &seq1, const DevArray &seq2, int block_count, int seq_index) :
            seq1(seq1), seq2(seq2), block_count(block_count), seq_index(seq_index) {}

};

struct Block {
    WorkUnit workUnit;
    SharedVars sharedVars;

    Block(const WorkUnit &workUnit, const SharedVars &sharedVars) :
            workUnit(workUnit), sharedVars(sharedVars) {}
};

struct DevArray {
    //inclusive
    int start;
    //exclusive
    int end;
    int size = end - start; //TODO check
    const CUdeviceptr array;

    DevArray(CUdeviceptr array, int start, int end) : array(array), start(start), end(end) {}
};

struct WorkUnit {
    DevArray seq;
    int pivot;

    WorkUnit(DevArray seq, int pivot) : seq(seq), pivot(pivot) {}

};

#endif //SORTS_QUICK_SORT_H
