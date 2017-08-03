
#include "quick_sort.h"
#include "../utils/cuda_device.h"

extern "C" {


//__global__ int s_start;
//__global__ int s_end;
//__global__ int old_start;
//__global__ int old_end;
//__global__ int block_count;

__device__
int median(int* array, int i1, int i2, int i3) {
    int tab[3] = {array[i1], array[i2], array[i3]};
    //improve
    for (int i=0; i< 3; ++i) {
        for (int j=0; j< i; ++j) {
            int max = array[i];
            int min = array[j];
            if (array[i] < array[j]) {
                int key = array[i];
                array[i] = array[j];
                array[j] = key;
            }
        }
    }
    return tab[1];
}

__device__
void pref_sum(int* array) {
    __shared__ int shared[2][QUICK_THREADS_IN_BLOCK];

    shared[0][threadIdx.x] = array[threadIdx.x];
    __syncthreads();

    bool from = 1;
    bool to = 0;
    for (int d = 1; d < QUICK_THREADS_IN_BLOCK * 2; d <<= 1) {
        from = !from;
        to = !to;
        if (2 * threadIdx.x >= d) {
            shared[to][2 * threadIdx.x] = shared[from][2 * threadIdx.x - d] + shared[from][2 * threadIdx.x];
        } else {
            shared[to][2 * threadIdx.x] = shared[from][2 * threadIdx.x];
        }

        if (2 * threadIdx.x + 1 >= d) {
            shared[to][2 * threadIdx.x + 1] = shared[from][2 * threadIdx.x + 1 - d] + shared[from][2 * threadIdx.x + 1];
        } else {
            shared[to][2 * threadIdx.x + 1] = shared[from][2 * threadIdx.x + 1];
        }
        __syncthreads();
    }

    array[threadIdx.x] = shared[to][threadIdx.x];
    //sync?
    __syncthreads();
}

__global__
void gqsort(Block* blocks, int* in, int* out, WorkUnit* news, int sequences) {
    //cached in to shared ?
    __shared__ int lt[QUICK_THREADS_IN_BLOCK],
            gt[QUICK_THREADS_IN_BLOCK],
            pivot,
            start,
            end,
    lbeg, gbeg;
//    __shared__ int* s;
//    __shared__ int* out_s;
    int i,l_from, g_from, l_pivot, g_pivot;

    int blockId = blockIdx.x + blockIdx.y* gridDim.x;
    int gthid = threadIdx.x + blockId * blockDim.x;

    SharedVars* parent;

    if (threadIdx.x ==0) {
        //broadcast?
        //sync needed?
        Block block = blocks[blockId];
        parent = &blocks[blockId].sharedVars;
//        s = (int*) block.workUnit.seq.array;
        start = block.workUnit.seq.start;
        end = block.workUnit.seq.end;
        pivot = block.workUnit.pivot;
    }


    lt[threadIdx.x] = 0;
    gt[threadIdx.x] = 0;

   i = start + threadIdx.x;

    for (; i< end; i += QUICK_THREADS_IN_BLOCK) {
        if (in[i] < pivot) lt[threadIdx.x]++;
        if (in[i] > pivot) gt[threadIdx.x]++;
    }
    pref_sum(lt);
    pref_sum(gt);


    if (threadIdx.x == QUICK_THREADS_IN_BLOCK - 1) {
        lbeg = atomicAdd(&parent->seq1.start, lt[QUICK_THREADS_IN_BLOCK - 1]);
        gbeg = atomicSub(&parent->seq1.start, gt[QUICK_THREADS_IN_BLOCK - 1]);
    }
    l_from = lbeg + lt[threadIdx.x];
    g_from = gbeg + gt[threadIdx.x];

    i = start + threadIdx.x;
    for (; i< end; i+=QUICK_THREADS_IN_BLOCK) {
        if (in[i] < pivot) {
            out[l_from++] = in[i];
        }
        if (in[i] > pivot) {
            out[g_from++] = in[i];
        }
    }
    if (threadIdx.x == 0 && atomicSub(&parent->block_count, 1)) {
        for(i = parent->seq1.start, i < parent->seq1.end;i++) {
            in[i] = pivot;
        }
        int old_start = parent->seq2.start;
        int old_end = parent->seq2.end-1;
        int s_start = parent->seq1.start;
        int s_end = parent-> seq1.end -1;
        l_pivot = median(out, old_start,(old_start + s_start)/2,  s_start);
        g_pivot = median(out,s_end, (s_end + old_end)/2, old_end );

        news[parent->seq_index] = WorkUnit(DevArray((CUdeviceptr) out, old_start, s_start), l_pivot);
        news[parent->seq_index] = WorkUnit(DevArray((CUdeviceptr) out, s_end, old_end), g_pivot);
    }
};

__global__ lqsort() {

};

}