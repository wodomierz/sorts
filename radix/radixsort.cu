#include <cstdio>
#include "../utils/cuda_device.h"
#include "../prefsum/prefsum.cuh"
extern "C" {
__global__
void prefixSum(int* in , int* out, int* prefixSums, int size, int mask) {
  __shared__ int shared[2][2048];
	int thid = 2*((blockIdx.x *  blockDim.x) + threadIdx.x);

	if (thid >= size) {
		shared[0][2*threadIdx.x] = 0;
	}
	else {
		shared[0][2*threadIdx.x] = (in[thid] >> mask)&1;
	}

	if (thid + 1>= size) {
		shared[0][2*threadIdx.x + 1] = 0;
	}
	else {
		shared[0][2*threadIdx.x + 1] = (in[thid+1] >> mask)&1;
	}

	__syncthreads();

	bool to = 0;
	prefixSumDev<1024, 2>(shared, to);

	if (thid < size) out[thid] = shared[to][2*threadIdx.x];
	if (thid +1 < size) out[thid+1] = shared[to][2*threadIdx.x +1];

	if (2*threadIdx.x +1 == (2048 - 1)) {
		prefixSums[blockIdx.x + 1] = shared[to][2*threadIdx.x +1];
	}
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



