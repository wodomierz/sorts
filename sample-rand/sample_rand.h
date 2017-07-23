//
// Created by Miron Ficak on 09/07/2017.
//


#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_THREAD 8
#define S_SIZE 1024
#define MAX_SAMPLE_SIZE 131072
#define S_POW 10
#define M 2048
#define T THREADS_PER_BLOCK
#define L ELEMENTS_PER_THREAD
#define MAX_GRID_DIM 32768
#define K S_SIZE

#ifndef SORTS_SAMPLE_RAND_H
#define SORTS_SAMPLE_RAND_H
void sample_rand(int*, int);

#endif //SORTS_SAMPLE_RAND_H
