//
// Created by Miron Ficak on 09/07/2017.
//


#include <climits>
#include <cstdio>

#define ELEMENTS_PER_THREAD 4
#define S_POW 3
#define S_SIZE (1 << S_POW)
#define THREADS_PER_BLOCK (S_SIZE)

#define MAX_SAMPLE_SIZE 131072


#define T THREADS_PER_BLOCK
#define L ELEMENTS_PER_THREAD
#define ARRAYS_NUM 8
#define A ARRAYS_NUM
#define BLOCK_SIZE (T * L)

#define M BLOCK_SIZE

#define PREFSUM_THREADS 1024
#define PREFSUM_ELEMENTS_PER_THREAD 2
#define PREFSUM_BLOCK_SIZE (PREFSUM_THREADS * PREFSUM_ELEMENTS_PER_THREAD)

#define K S_SIZE

#ifndef SORTS_SAMPLE_RAND_H
#define SORTS_SAMPLE_RAND_H
void sampleRand(int*, int);

#endif //SORTS_SAMPLE_RAND_H
