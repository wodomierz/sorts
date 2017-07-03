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




