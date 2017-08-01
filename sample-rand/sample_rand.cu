#include "sample_rand.h"

extern "C" {

__device__
void min_max(int *tab, int for_min, int for_max, int size) {
    if (for_min >= size || for_max >= size) {
        return;
    }
    int min = tab[for_min];
    int max = tab[for_max];
    if (max < min) {
        atomicExch(tab + for_max, min);
        atomicExch(tab + for_min, max);
    }
};


__global__
void chujowy_sort(int * to_sort, int size) {
    for (int i=1; i< size; ++i) {
        for (int j=0; j < i; ++j) {
            min_max(to_sort, j, i, size);
            __syncthreads();
        }
    }
}



__global__
void odd_even(int *to_sort) {
    //TODO you MUST check size
    __shared__ int tab[T*2];

    int x = blockIdx.x * blockDim.x *2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = threadIdx.x;
    int gthid = x + y * gridDim.x * blockDim.x * 2;
    //TODO check *2 here

    tab[thid] = to_sort[gthid];
    tab[thid + T] = to_sort[gthid + T];
    __syncthreads();

    for (int pow__half_batch = 0, half_batch = 1;
         pow__half_batch <= 10;
         pow__half_batch++, half_batch <<=1) {

        int wireThid = thid + ((thid >> pow__half_batch) << pow__half_batch);
        int opposite = wireThid + half_batch;
        min_max(tab, wireThid, opposite, T*2);
        __syncthreads();
        for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {

            int d = 1 << d_power;

            int period = half_batch - d;

            int wire_id = thid + (((thid>>d_power) + ((thid / period) << 1) + 1) << d_power);
            int opposite = wire_id + d;
            min_max(tab, wire_id, opposite, T*2);

            __syncthreads();
        }

    }

    to_sort[gthid] = tab[thid];
    to_sort[gthid + T] = tab[thid + T];

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
void counters(int *to_sort, int *sample, int *prefsums, int number_of_blocks) {
    __shared__ int bst[S_SIZE];
    __shared__ int histogram[S_SIZE];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gthid = x + y * gridDim.x * blockDim.x;

    int threadId = threadIdx.x;

    if (threadId < S_SIZE) { //?
        bst[threadId] = sample[threadId];
        histogram[threadId] = 0;
    }
    __syncthreads();


    int j = findIndex(to_sort[gthid], bst);
    atomicAdd(histogram + j, 1);
    __syncthreads();


    if (threadId < S_SIZE) {
        //bug?
        int index = (threadId * number_of_blocks) + blockIdx.x;
        atomicExch(prefsums + index, histogram[threadId]);
    }
}

__global__
void prefsum1(int *localPrefsums, int *maxPrefSums, int number_of_blocks, int* sample_offsets ) {
    int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = x + y * gridDim.x * blockDim.x * 2;

    int global_offset = maxPrefSums[blockIdx.x];

    atomicAdd(localPrefsums + thid, global_offset);
    atomicAdd(localPrefsums + thid + THREADS_PER_BLOCK, global_offset);

    //czy ta atomiczność i synchronizacja są w ogóle potrzebne?
    __syncthreads();
    if (thid % number_of_blocks == number_of_blocks -1) {
        atomicExch(sample_offsets + thid/number_of_blocks + 1, localPrefsums[thid]);
    }
    __syncthreads();
    if ((thid + T) % number_of_blocks == number_of_blocks - 1) {
        atomicExch(sample_offsets + (thid+ T)/number_of_blocks + 1, localPrefsums[thid + T]);
    }


}
__global__
void prefsum(int *localPrefsums, int *maxPrefSums) {
    __shared__ int shared[2][T*2 + 1];

    int x = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = x + y * gridDim.x * blockDim.x * 2;

    shared[0][threadIdx.x] = localPrefsums[thid];
    shared[0][threadIdx.x + T] = localPrefsums[thid + T];

    __syncthreads();

    bool from = 1;
    bool to = 0;
    for (int d = 1; d < T*2; d <<= 1) {
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

    localPrefsums[thid] = shared[to][threadIdx.x];
    localPrefsums[thid + T] = shared[to][threadIdx.x + T];
    if (2 * threadIdx.x + 1 == (T*2 - 1)) {
        maxPrefSums[blockIdx.x + 1] = shared[to][2 * threadIdx.x + 1];
    }

}

__global__
void scatter(int *in, int *out, int *sample, int *prefsums, int number_of_blocks) {
    __shared__ int bst[S_SIZE];
    __shared__ int histogram[S_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    int gthid = x + y * gridDim.x * blockDim.x;

    int threadId = threadIdx.x;

    if (threadId < S_SIZE) { //?
        bst[threadId] = sample[threadId];
        histogram[threadId] = 0;
    }
    __syncthreads();

    int e = in[gthid];
    int j = findIndex(e, bst);


    int local_index = atomicAdd(histogram + j, 1);
    __syncthreads();

    int indexInPrefsums = (j * number_of_blocks) + blockId;
    int offset = 0;
    if (indexInPrefsums > 0 ) {
        offset = prefsums[indexInPrefsums -1];
    }
    out[offset + local_index] = e;

}

}