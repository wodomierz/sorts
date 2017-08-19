#include <cstdio>
#include "../utils/cuda_device.h"
#include "../prefsum/prefsum.cuh"
#include "../utils/kernel_commons.cuh"



template <int Threads, int Elements>
__device__ __forceinline__
void radix_pref_sum_dev(int* in , int* out, int* prefixSums, int size, int mask) {
	int const BlockSize = Threads * Elements;
  __shared__ int shared[2][BlockSize];

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int offset = blockId * BlockSize;

	in += offset;
	out += offset;
	size -=offset;

	for (int thid = threadIdx.x; thid < BlockSize; thid += Threads) {
		shared[0][thid] = thid < size ? (in[thid] >> mask)&1 : 0;
	}
	__syncthreads();

	bool to = 0;
	prefixSumDev<Threads, Elements>(shared, to);

	for (int thid = threadIdx.x; thid < BlockSize && thid < size; thid += Threads) {
		out[thid] = shared[to][thid];
	}
	if (threadIdx.x == Threads -1) {
		prefixSums[blockId + 1] = shared[to][BlockSize - 1];
	}
}

extern "C" {
__global__
void prefixSum(int* in , int* out, int* prefixSums, int size, int mask) {
	radix_pref_sum_dev<1024, 2>(in, out, prefixSums, size, mask);
}

__global__
void sort(int* source, int* destination, int* prefixSum, int* prefixSums, int mask, int n,  int number_of_zeros) {
	int thid =2*(blockIdx.x *  blockDim.x) + threadIdx.x;
		if (thid < n) {
			int place;
			if ((source[thid]>>mask)&1) {
				place = number_of_zeros + prefixSum[thid] + prefixSums[blockIdx.x] - 1;
			}
			else {
				place = thid - prefixSum[thid] - prefixSums[blockIdx.x];
			}
			destination[place] = source[thid];
		}
		thid += 1024;
		if(thid < n) {
			int place;
			if ((source[thid]>>mask)&1) {
				place = number_of_zeros + prefixSum[thid] + prefixSums[blockIdx.x] - 1;
			}
			else {
				place = thid - prefixSum[thid] - prefixSums[blockIdx.x];
			}
			destination[place] = source[thid];
		}
}
}



