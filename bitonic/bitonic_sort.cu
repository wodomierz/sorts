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
void bitonic_merge1(int *to_sort) {
    __shared__ int tab[1024];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gthid = x + y * gridDim.x * blockDim.x;

    int threadId = threadIdx.x;
//    int gthid = blockIdx.x * blockDim.x + threadId;
//    int y = blockIdx.y * blockDim.y + threadIdx.y;
    tab[threadId] = to_sort[gthid];

    for (int d_triangle = 2, d_half_traingle_p = 0;
         d_half_traingle_p <= 9;
         d_half_traingle_p++, d_triangle<<=1) {

        int wireThid = threadId + ((threadId >> d_half_traingle_p) << d_half_traingle_p);
        int local_thid = wireThid & (d_triangle-1);
        int opposite = wireThid - local_thid + d_triangle - 1 - local_thid;
        min_max(tab, wireThid, opposite, 1024);
        __syncthreads();
//        manageResult(cuLaunchKernel(bitonic_triangle_merge, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args1, 0),"running");
//        cuCtxSynchronize();
        for (int d_power = d_half_traingle_p -1; d_power >= 0; d_power--) {

            int wireThid = threadId + ((threadId >> d_power) << d_power);

            int opposite = wireThid + (1 << d_power);
            min_max(tab, wireThid, opposite, 1024);

//            void* args2[3] = { &deviceToSort, &d_p, &size};
//            manageResult(cuLaunchKernel(bitonic_merge, x_dim, y_dim, 1, THREADS_IN_BLOCK, 1, 1, 0, 0, args2, 0),"running");
            __syncthreads();


        }
    }
    to_sort[gthid] = tab[threadId];
//    tab[threadId];

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


