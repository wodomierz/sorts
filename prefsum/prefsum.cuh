#ifndef SORTS_PREFSUM_CUH
#define SORTS_PREFSUM_CUH

#include "../utils/cuda_device.h"

template <int THREADS, int ELEMENTS>
__device__ __forceinline__
void prefixSumDev(int (&shared)[2][ELEMENTS*THREADS], bool& to) {
    const int BLOCK = THREADS*ELEMENTS;

    bool from = 1;
    to = 0;
    for (int d = 1; d < BLOCK; d<<=1) {
        from = !from;
        to = !to;
        for (int i= 0; i< ELEMENTS; ++i) {
            int thid = threadIdx.x + i*THREADS;
            if (thid >= d) {
                shared[to][thid] = shared[from][thid - d] + shared[from][thid];
            }
            else {
                shared[to][thid] = shared[from][thid];
            }

        }
        __syncthreads();
    }
}

template <int Threads, int Elements>
__device__ __forceinline__
void pref_sum_and_move_result_dev(int (&shared)[2][Threads*Elements], int* out, int* block_sums, int size, int blockId) {
    const int BlockSize = Threads * Elements;
    bool to = 0;
    prefixSumDev<Threads, Elements>(shared, to);

    for (int thid = threadIdx.x; thid < BlockSize && thid < size; thid += Threads) {
        out[thid] = shared[to][thid];
    }
    if (threadIdx.x == Threads -1) {
        block_sums[blockId + 1] = shared[to][BlockSize - 1];
    }
};

#endif //SORTS_PREFSUM_CUH