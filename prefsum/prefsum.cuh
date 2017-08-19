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
