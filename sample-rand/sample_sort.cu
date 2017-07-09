
extern "C" {

__global__
void prefsum(int *to_sort, int *sample, int sample_size,int sample_pow, int size, int* prefsums) {
    __shared__ int bst[1024];
    __shared__ int histogram[1024];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int gthid = x + y * gridDim.x * blockDim.x;

    int threadId = threadIdx.x;

    if (threadId < sample_size) {
        bst[threadId] = sample[threadId];
        histogram[threadId] = 0;
    }
    __syncthreads();

    int e = to_sort[gthid];
    int j = 1;
    int k = sample_pow;
    while(k--) {
        j = 2*j +(e> bst[j]);
    }
    j = j-sample_size +1; // bucket index

}

}