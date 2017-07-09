#include "sample_rand.h"

extern "C" {

__device__ int findIndex(int e) {
    int j = 1;
    int k = S_POW;
    while(k--) {
        j = 2*j +(e> bst[j]);
    }
    j = j-S_SIZE +1; // bucket index
    return j;
}

__global__
void counters(int *to_sort, int *sample, int size, int* prefsums, int number_of_blocks) {
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

    int j = findIndex(to_sort[gthid]);
    atomicAdd(histogram + j, 1);
    __syncthreads();

    if (threadId < S_SIZE) {
        //bug?
        int index = (threadId * number_of_blocks) + blockIdx.x;
        prefsums[index] = histogram[threadId];
    }
}

__global__
void prefsum1(int* localPrefsums, int*maxPrefSums) {
    int x = blockIdx.x * blockDim.x*2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = x + y * gridDim.x * blockDim.x*2;

    int global_offset = maxPrefSums[blockIdx.x];

    atomicAdd(localPrefsums + thid, global_offset);
    atomicAdd(localPrefsums + thid +1024, global_offset);

//    __synchthreads();

}
__global__
void prefsum(int* localPrefsums, int* maxPrefSums) {
    __shared__ int shared[2][2048];

    int x = blockIdx.x * blockDim.x*2 + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int thid = x + y * gridDim.x * blockDim.x*2;

    shared[0][threadIdx.x] = localPrefsums[thid];
    shared[0][threadIdx.x + 1024] = localPrefsums[thid + 1024];

    __synchthreads();

    bool from = 1;
    bool to = 0;
    for (int d = 1; d < 2048; d<<=1) {
        from = !from;
        to = !to;
        if (2*threadIdx.x >= d) {
            shared[to][2*threadIdx.x] = shared[from][2*threadIdx.x - d] + shared[from][2*threadIdx.x];
        }
        else {
            shared[to][2*threadIdx.x] = shared[from][2*threadIdx.x];
        }

        if(2*threadIdx.x + 1 >= d ){
            shared[to][2*threadIdx.x + 1] = shared[from][2*threadIdx.x +1 - d] + shared[from][2*threadIdx.x +1];
        }
        else {
            shared[to][2*threadIdx.x +1 ] = shared[from][2*threadIdx.x +1];
        }

        __syncthreads();
    }

    localPrefsums[thid] = shared[to][threadIdx.x];
    localPrefsums[thid+1024] = shared[to][threadIdx.x +1024];

    if (2*threadIdx.x +1 == (2048 - 1)) {
        maxPrefSums[blockIdx.x + 1] = shared[to][2*threadIdx.x +1];
    }

}

__global__
void scatter(int *in, int * out,  int *sample, int size, int* prefsums, int number_of_blocks) {
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

    int e =in[gthid];
    int j = findIndex(in, gthid);


    int local_index= atomicAdd(histogram + j, 1);
    __syncthreads();

    int indexInPrefsums = (j * number_of_blocks) + blockIdx.x;
    out[prefsums[indexInPrefsums] + local_index] = e;
//    if (threadId < S_SIZE) {
//        int index = (j * number_of_blocks) + blockIdx.x;
//        prefsums[index] = histogram[threadId];
//    }
}

}