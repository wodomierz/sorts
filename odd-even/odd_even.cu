//#include "../utils/kernel_commons.cuh"

extern "C" {

__device__
static int THREADS_IN_BLOCK = 1024;

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
} ;

__global__
void odd_even(int *to_sort) {
	__shared__ int tab[2048];

	int x = blockIdx.x * blockDim.x *2 + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int thid = threadIdx.x;
	int gthid = x + y * gridDim.x * blockDim.x * 2;
	//TODO check *2 here

	tab[thid] = to_sort[gthid];
	tab[thid + 1024] = to_sort[gthid + 1024];
	__syncthreads();

	for (int pow__half_batch = 0, half_batch = 1;
		 pow__half_batch <= 10;
		 pow__half_batch++, half_batch <<=1) {

		int wireThid = thid + ((thid >> pow__half_batch) << pow__half_batch);
		int opposite = wireThid + half_batch;
		min_max(tab, wireThid, opposite, 2048);
		__syncthreads();
		for (int d_power = pow__half_batch - 1; d_power >= 0; d_power--) {

			int d = 1 << d_power;

			int period = half_batch - d;

			int wire_id = thid + (((thid>>d_power) + ((thid / period) << 1) + 1) << d_power);
			int opposite = wire_id + d;
			min_max(tab, wire_id, opposite, 2048);

			__syncthreads();
		}

	}

	to_sort[gthid] = tab[thid];
	to_sort[gthid + 1024] = tab[thid + 1024];

}

__global__
void odd_even_phase1(int *to_sort, int d_pow, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thid = x + y * gridDim.x * blockDim.x;

	int wireThid = thid + ((thid >> d_pow) << d_pow);

	int opposite = wireThid + (1 << d_pow);
	min_max(to_sort, wireThid, opposite, size);
}


__global__
void odd_even_phase2(int *to_sort, int d_power, int half_batch_size, int size) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int thid = x + y * gridDim.x * blockDim.x;

	int d = 1 << d_power;

	int period = half_batch_size - d;

	int wire_id = thid + (((thid>>d_power) + ((thid / period) << 1) + 1) << d_power);
	int opposite = wire_id + d;
	min_max(to_sort, wire_id, opposite, size);

}

}




