//
// Created by Miron Ficak on 05/08/2017.
//
#include "cuda.h"
#ifndef SORTS_QUICK_SHARED_H
#define SORTS_QUICK_SHARED_H

//#include "../utils/cuda_device.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __device__
#else
#define CUDA_HOSTDEV
#endif

struct DevArray {
    //inclusive
    int start;
    //exclusive
    int end;
    int size;
//        = end -start;
    //TODO check

    CUDA_HOSTDEV DevArray( int start, int end): start(start), end(end) {};
    CUDA_HOSTDEV DevArray() {};

};
//typedef struct DevArray DevArray;

struct SharedVars {
    int seq_index;
    DevArray seq1;
    DevArray seq2;
    int block_count;

    CUDA_HOSTDEV SharedVars(const DevArray seq1, const DevArray seq2, int block_count, int seq_index) :
    seq1(seq1), seq2(seq2), block_count(block_count), seq_index(seq_index) {}

};
typedef struct SharedVars SharedVars;

struct WorkUnit {
    DevArray seq;
    int pivot;

    CUDA_HOSTDEV WorkUnit(DevArray seq, int pivot) : seq(seq), pivot(pivot) {}

};
typedef struct WorkUnit WorkUnit;

struct Block {
    WorkUnit workUnit;
    SharedVars sharedVars;

    CUDA_HOSTDEV Block(const WorkUnit workUnit, const SharedVars sharedVars) :
    workUnit(workUnit), sharedVars(sharedVars) {}
};


#endif //SORTS_QUICK_SHARED_H
