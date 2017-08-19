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

template <int Threads, int Elements>
__device__ __forceinline__
void sort_dev(int* source, int* destination, int* prefixSum, int* prefixSums, int mask, int n,  int number_of_zeros) {
	const int BlockSize = Threads * Elements;
	int blockId = one_dimension_blockId();
	int offset = blockId * BlockSize;
	prefixSum += offset;
	source += offset;
	n -= offset;
	for (int thid = threadIdx.x; thid < BlockSize && thid < n; thid += Threads) {
		int place;
		if ((source[thid]>>mask)&1) {
			place = number_of_zeros + prefixSum[thid] + prefixSums[blockId] - 1;
		}
		else {
			place = offset + thid - prefixSum[thid] - prefixSums[blockId];
		}
		destination[place] = source[thid];
	}

}



extern "C" {
__global__
void prefixSum(int* in , int* out, int* prefixSums, int size, int mask) {
	radix_pref_sum_dev<1024, 2>(in, out, prefixSums, size, mask);
}

__global__
void sort(int* source, int* destination, int* prefixSum, int* prefixSums, int mask, int n,  int number_of_zeros) {
	sort_dev<1024, 2>(source, destination, prefixSum, prefixSums, mask, n, number_of_zeros);
}
}



