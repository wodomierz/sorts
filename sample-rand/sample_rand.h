//
// Created by Miron Ficak on 09/07/2017.
//


#include <climits>
#include <cstdio>

#define ELEMENTS_PER_THREAD 1
#define S_POW 3
#define S_SIZE (1 << S_POW)
#define THREADS_PER_BLOCK (S_SIZE*2)

#define MAX_SAMPLE_SIZE 131072

#define M THREADS_PER_BLOCK
#define T THREADS_PER_BLOCK
#define L ELEMENTS_PER_THREAD
#define MAX_GRID_DIM (1 << 30)
#define BLOCK_SIZE (T * L)

#define PREFSUM_THREADS T
#define PREFSUM_ELEMENTS_PER_THREAD 2
#define PREFSUM_BLOCK_SIZE (PREFSUM_THREADS * PREFSUM_ELEMENTS_PER_THREAD)

#define K S_SIZE

#ifndef SORTS_SAMPLE_RAND_H
#define SORTS_SAMPLE_RAND_H
void sample_rand(int*, int);

#endif //SORTS_SAMPLE_RAND_H
