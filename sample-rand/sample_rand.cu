#include "sample_rand.h"



#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"

extern "C" {


__device__ __forceinline__
int place(int size, int seed, int plus, int i) {
    return (i * seed + plus) % size;
}


__global__
void chujowy_sort(int *to_sort, int size) {
    for (int i = 1; i < size; ++i) {
        for (int j = 0; j < i; ++j) {
            min_max(to_sort, j, i, size);
            __syncthreads();
        }
    }
}

__device__
void chujowy_sort1(int *to_sort, int size) {
    __syncthreads();
    if (threadIdx.x ==0) {
        for (int i = 1; i < size; ++i) {
            for (int j = 0; j < i; ++j) {
                min_max(to_sort, j, i, size);
            }
        }
    }
    __syncthreads();
}


__global__
void odd_even(int *to_sort) {
    //TODO you MUST check size
    __shared__ int tab[THREADS_PER_BLOCK * 2];

    int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = threadIdx.x;
    int gthid = x + y * gridDim.x * blockDim.x * 2;
    //TODO check *2 here

    tab[thid] = to_sort[gthid];
    tab[thid + THREADS_PER_BLOCK] = to_sort[gthid + THREADS_PER_BLOCK];
    __syncthreads();

    for (int pow__half_batch = 0, half_batch = 1;
         pow__half_batch <= 10;
         pow__half_batch++, half_batch <<= 1) {

        int wireThid = thid + ((thid >> pow__half_batch) << pow__half_batch);
        int opposite = wireThid + half_batch;
        min_max(tab, wireThid, opposite, THREADS_PER_BLOCK * 2);
        __syncthreads();
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {

            int d = 1 << d_power;

            int period = half_batch - d;

            int wire_id = thid + (((thid >> d_power) + ((thid / period) << 1) + 1) << d_power);
            int opposite = wire_id + d;
            min_max(tab, wire_id, opposite, THREADS_PER_BLOCK * 2);

            __syncthreads();
        }

    }

    to_sort[gthid] = tab[thid];
    to_sort[gthid + THREADS_PER_BLOCK] = tab[thid + THREADS_PER_BLOCK];

}

__global__
void sample(int *tab, int size, int seed, int plus, int *bst) {
    __shared__ int to_sort[SAMPLE_BLOCK];
    int thid = threadIdx.x;

    int i = thid;
    for (;i < SAMPLE_BLOCK; i+= SAMPLE_THREADS) {
        to_sort[i] = tab[place(size, seed, plus, i)];
    }

    __syncthreads();

    chujowy_sort1(to_sort, SAMPLE_BLOCK);
    __syncthreads();
    if (threadIdx.x ==0) {
        int iteratr = 0;
        for (int i = 2; i <= S_SIZE; i *= 2) {
            for (int j = 1; j < i; j += 2) {
                bst[iteratr++] = to_sort[(j * (S_SIZE / i) - 1)*AS];
            }
        }
        bst[S_SIZE - 1] = 0;
    }
}


__device__ int findIndex(int e, int *bst) {
    int j = 1;
    int k = S_POW;
    while (k--) {
        j = 2 * j + (e > bst[j - 1]);
    }
    j = j - S_SIZE; // bucket index
    return j;
}


__global__
void prefsum1(int *localPrefsums, int *maxPrefSums, int number_of_blocks, int *sample_offsets, int size) {
    int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = x + y * gridDim.x * blockDim.x * 2;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int global_offset = maxPrefSums[blockId];

    for (int i = 0; i < 2 && thid + i * PREFSUM_THREADS < size; ++i) {
        localPrefsums[thid + i * PREFSUM_THREADS] += global_offset;
        if ((thid + i * PREFSUM_THREADS) % number_of_blocks == number_of_blocks - 1) {
            sample_offsets[(thid + i * PREFSUM_THREADS) / number_of_blocks + 1] = localPrefsums[thid +
                                                                                                i * PREFSUM_THREADS];
        }
    }


}

__device__
int getOrZero(int *tab, int i, int size) {
    return i < size ? tab[i] : 0;
}
__global__
void prefsum(int *localPrefsums, int *maxPrefSums, int size) {
    __shared__ int shared[2][PREFSUM_THREADS * 2 + 1];

    int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int y = blockIdx.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int thid = x + y * gridDim.x * blockDim.x * 2;

    shared[0][threadIdx.x] = getOrZero(localPrefsums, thid, size);
    shared[0][threadIdx.x + PREFSUM_THREADS] = getOrZero(localPrefsums, thid + PREFSUM_THREADS, size);
//    shared[0][threadIdx.x] = localPrefsums[thid];
//    shared[0][threadIdx.x + PREFSUM_THREADS] = localPrefsums[thid + PREFSUM_THREADS];
    __syncthreads();

    bool from = 1;
    bool to = 0;
    for (int d = 1; d < PREFSUM_THREADS * 2; d <<= 1) {
        from = !from;
        to = !to;
        if (2 * threadIdx.x >= d) {
            shared[to][2 * threadIdx.x] = shared[from][2 * threadIdx.x - d] + shared[from][2 * threadIdx.x];
        } else {
            shared[to][2 * threadIdx.x] = shared[from][2 * threadIdx.x];
        }

        if (2 * threadIdx.x + 1 >= d) {
            shared[to][2 * threadIdx.x + 1] = shared[from][2 * threadIdx.x + 1 - d] + shared[from][2 * threadIdx.x + 1];
        } else {
            shared[to][2 * threadIdx.x + 1] = shared[from][2 * threadIdx.x + 1];
        }
        __syncthreads();
    }

    if (thid < size) localPrefsums[thid] = shared[to][threadIdx.x];
    if (thid + PREFSUM_THREADS < size)
        localPrefsums[thid + PREFSUM_THREADS] = shared[to][threadIdx.x + PREFSUM_THREADS];
    if (2 * threadIdx.x + 1 == (PREFSUM_THREADS * 2 - 1)) {
        maxPrefSums[blockId + 1] = shared[to][2 * threadIdx.x + 1];
    }

}

__global__
void counters(int *to_sort, int *sample, int *prefsums, int number_of_blocks, int size) {
    __shared__ int bst[S_SIZE];
    __shared__ int histogram[A][S_SIZE];


    int x = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int gthid = x + y * gridDim.x * blockDim.x * ELEMENTS_PER_THREAD;

    int threadId = threadIdx.x;

    if (threadId < S_SIZE) { //?
        bst[threadId] = sample[threadId];
        for (int a = 0; a < A; ++a) {
            histogram[a][threadId] = 0;
        }
    }
    __syncthreads();


    for (int i = 0; i < ELEMENTS_PER_THREAD && gthid + i * THREADS_PER_BLOCK < size; ++i) {
//    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        //ke?

        int j = findIndex(to_sort[gthid + i * THREADS_PER_BLOCK], bst);
        atomicAdd(histogram[gthid % A] + j, 1);
    }
    __syncthreads();


    if (threadId < S_SIZE) {
        //bug?
        for (int i = 1; i < A; ++i) {
            histogram[0][threadId] += histogram[i][threadId];
        }
        int index = (threadId * number_of_blocks) + blockId;
        atomicExch(prefsums + index, histogram[0][threadId]);
    }
}

__global__
void scatter(int *in, int *out, int *sample, int *prefsums, int number_of_blocks, int size) {
    __shared__ int bst[S_SIZE];
    __shared__ int histogram[S_SIZE];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int gthid = x + y * gridDim.x * BLOCK_SIZE;
    int threadId = threadIdx.x;
    if (threadId < S_SIZE) { //?
        bst[threadId] = sample[threadId];
        histogram[threadId] = 0;
    }
    __syncthreads();

    for (int i = 0; i < ELEMENTS_PER_THREAD && gthid + i * THREADS_PER_BLOCK < size; ++i) {
//    for (int i=0; i < ELEMENTS_PER_THREAD; ++i) {
        //ke?
        int e = in[gthid + i * THREADS_PER_BLOCK];
        int j = findIndex(e, bst);
        int local_index = atomicAdd(histogram + j, 1);
        int indexInPrefsums = (j * number_of_blocks) + blockId;
        int offset = 0;
        if (indexInPrefsums > 0) {
            offset = prefsums[indexInPrefsums - 1];
        }
        out[offset + local_index] = e;
    }
}

}