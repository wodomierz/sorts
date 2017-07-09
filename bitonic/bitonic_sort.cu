//#include <cstdio>


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
} ;

__global__
void bitonic_merge2(int *to_sort) {
    __shared__ int tab[2048];
    int x = 2*blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gthid = x + y * gridDim.x * blockDim.x*2;

    int threadId = threadIdx.x;

    tab[threadId] = to_sort[gthid];
    tab[threadId + 1024] = to_sort[gthid + 1024];

    __syncthreads();

    for (int d_triangle = 2, d_half_traingle_p = 0;
         d_half_traingle_p <= 10;
         d_half_traingle_p++, d_triangle<<=1) {

        int wireThid = threadId + ((threadId >> d_half_traingle_p) << d_half_traingle_p);

        int local_thid = wireThid & (d_triangle-1);
        int opposite = wireThid - local_thid + d_triangle - 1 - local_thid;

        min_max(tab, wireThid, opposite, 2048);
        __syncthreads();
        for (int d_power = d_half_traingle_p -1; d_power >= 0; d_power--) {
            int wireThid = threadId + ((threadId >> d_power) << d_power);
            int opposite = wireThid + (1 << d_power);
            min_max(tab, wireThid, opposite, 2048);
            __syncthreads();
        }
    }
    to_sort[gthid] = tab[threadId];
    to_sort[gthid + 1024] = tab[threadId + 1024];
}

__global__
void bitonic_merge(int *to_sort, int d_power, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;

    int wireThid = thid + ((thid >> d_power) << d_power);

    int opposite = wireThid + (1 << d_power);
    min_max(to_sort, wireThid, opposite, size);
}

__global__
void bitonic_triangle_merge(int *to_sort, int half_triangle_power, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int thid = x + y * gridDim.x * blockDim.x;

    int d_triangle = 1 << (half_triangle_power + 1);
    int wireThid = thid + ((thid >> half_triangle_power) << half_triangle_power);
    int local_thid = wireThid & ( d_triangle-1);
    int opposite = wireThid - local_thid + d_triangle - 1 - local_thid;
    min_max(to_sort, wireThid, opposite, size);
}

}


