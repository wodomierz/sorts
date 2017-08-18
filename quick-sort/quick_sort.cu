
#include "quick_sort.h"
#include "../utils/cuda_device.h"
#include "../utils/kernel_commons.cuh"
#include "quick_shared.h"
#include "../bitonic/bitonic_sort.cuh"


extern "C" {

__device__
void alt_sort(DevArray array, int *out) {
    for (int i = array.start +1; i < array.end; ++i) {
        for (int j = array.start; j < i; ++j) {
            min_max(out,  j, i, array.end);
        }
    }
//    for (int i = array.start; i < array.end; ++i) {
//        out[i] = in[i];
//    }
}



__device__ __forceinline__
int median(int *array, int start, int end) {
    int tab[3] = {array[start], array[(start + end -1)/2], array[end -1]};
    //improve
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) {
            if (tab[i] < tab[j]) {
                int key = tab[i];
                tab[i] = tab[j];
                tab[j] = key;
            }
        }
    }
    return tab[1];
}

__global__
void pivot(int* array, int size, int* result) {
    *result = median(array, 0, size);
}

__device__ __forceinline__
void pref_sum(int shared[][QUICK_THREADS_IN_BLOCK], int *array) {
    shared[0][threadIdx.x] = array[threadIdx.x];
    __syncthreads();

    bool from = 1;
    bool to = 0;
    for (int d = 1; d < QUICK_THREADS_IN_BLOCK; d <<= 1) {
        from = !from;
        to = !to;
        if (threadIdx.x >= d) {
            shared[to][threadIdx.x] = shared[from][threadIdx.x - d] + shared[from][threadIdx.x];
        } else {
            shared[to][threadIdx.x] = shared[from][threadIdx.x];
        }
        __syncthreads();
    }

    array[threadIdx.x + 1] = shared[to][threadIdx.x];
    if (threadIdx.x == 0) {
        array[0] = 0;
    }
//    __syncthreads();
}

__global__
void gqsort(Block *blocks, int *in, int *out, WorkUnit *news) {
    //cached in to shared ?
    __shared__ int lt[QUICK_THREADS_IN_BLOCK + 1],
        gt[QUICK_THREADS_IN_BLOCK + 1],
        pivot,
        start,
        end,
        lbeg, gbeg;
    __shared__ int shared[2][QUICK_THREADS_IN_BLOCK];



    int i, l_from, g_from, l_pivot, g_pivot;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    SharedVars *parent;

    if (threadIdx.x == QUICK_THREADS_IN_BLOCK - 1) {
        //broadcast?
        //sync needed?
        Block block = blocks[blockId];
        parent = blocks[blockId].sharedVars;
        start = block.workUnit.seq.start;
        end = block.workUnit.seq.end;
        pivot = block.workUnit.pivot;

        lt[threadIdx.x + 1] =0;
        gt[threadIdx.x + 1] =0;
    }


    lt[threadIdx.x] = 0;
    gt[threadIdx.x] = 0;
    __syncthreads();
    i = start + threadIdx.x;

    for (; i < end; i += QUICK_THREADS_IN_BLOCK) {
        if (in[i] < pivot) lt[threadIdx.x]++;
        if (in[i] > pivot) gt[threadIdx.x]++;
    }
    pref_sum(shared, lt);
    pref_sum(shared, gt);


    if (threadIdx.x == QUICK_THREADS_IN_BLOCK - 1) {
        lbeg = atomicAdd(&(parent->seq.start), lt[QUICK_THREADS_IN_BLOCK]);
        gbeg = atomicSub(&(parent->seq.end), gt[QUICK_THREADS_IN_BLOCK]) - gt[QUICK_THREADS_IN_BLOCK];
    }
    __syncthreads();

    l_from = lbeg + lt[threadIdx.x];
    g_from = gbeg + gt[threadIdx.x];
    i = start + threadIdx.x;
    for (; i < end; i += QUICK_THREADS_IN_BLOCK) {
        if (in[i] < pivot) {
            out[l_from++] = in[i];
        }
        if (in[i] > pivot) {
            out[g_from++] =  in[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == QUICK_THREADS_IN_BLOCK - 1 && atomicSub(&(parent->block_count), 1) == 1) {
        for (i = parent->seq.start; i < parent->seq.end; i++) {
            out[i] =  pivot;
            in[i] = pivot;
        }
        l_pivot = median(out, parent->old_seq.start, parent->seq.start);
        g_pivot = median(out, parent->seq.end, parent->old_seq.end);

        news[2* parent->seq_index] = WorkUnit(DevArray(parent->old_seq.start, parent->seq.start), l_pivot);
        news[2* parent->seq_index + 1] = WorkUnit(DevArray(parent->seq.end, parent->old_seq.end), g_pivot);
    }
}




__device__ __forceinline__
void altOrPush(
    DevArray &devArray,
    DevArray* work_stack,
    int& stackIndex,
    int *out,
    int* tab) {
    if (devArray.end - devArray.start <= OTHER_SORT_LIM) {
        bitonic_merge_device<QUICKTHREADS_POW>(out + devArray.start, devArray.end - devArray.start, tab);
    } else {
        if (threadIdx.x ==0) work_stack[++stackIndex] = devArray;
    }
}



__global__
void lqsort(DevArray *seqs, int *in_h, int *out_h) {
    __shared__ int lt[QUICK_THREADS_IN_BLOCK + 1];
    __shared__ int gt[QUICK_THREADS_IN_BLOCK + 1];
    __shared__ int pivot;
    // how with shared memory ???
    __shared__ int start, end;
    __shared__ int gstart, gend;
    __shared__ DevArray workStack[1024];

    //TODO checkout
    __shared__ int workStcIndex;

    __shared__ int shared[2][QUICK_THREADS_IN_BLOCK];
    __shared__ int tab[QUICK_THREADS_IN_BLOCK *2];
    __shared__ DevArray long_seq, short_seq;

    int *out = out_h;
    int *in = in_h;
    int i, l_from, g_from;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    if (threadIdx.x == 0) {
        workStcIndex = 0;
        workStack[0] = seqs[blockId];
        gstart = workStack[0].start;
        gend = workStack[0].end;

    }
    __syncthreads();
    while (workStcIndex >= 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            DevArray dev = workStack[workStcIndex--];
            start = dev.start;
            end = dev.end;
            lt[QUICK_THREADS_IN_BLOCK] = 0;
            gt[QUICK_THREADS_IN_BLOCK] = 0;
            pivot = median(in, start, end);
        }
        lt[threadIdx.x] = 0;
        gt[threadIdx.x] = 0;
        __syncthreads();
        i = start + threadIdx.x;
        for (; i < end; i += QUICK_THREADS_IN_BLOCK) {
            if (in[i] < pivot) {
                lt[threadIdx.x]++;
            }
            if (in[i] > pivot) {
                gt[threadIdx.x]++;
            }
        }
        pref_sum(shared, lt); //exclusive
        __syncthreads();
        pref_sum(shared, gt);
        __syncthreads();
        l_from = start + lt[threadIdx.x];
        g_from = end - gt[threadIdx.x + 1];
        i = start + threadIdx.x;
        for (; i < end; i += QUICK_THREADS_IN_BLOCK) {
            if (in[i] < pivot) {
                out[l_from++] = in[i];
            }
            if (in[i] > pivot) {
                out[g_from++] = in[i];
            }
        }
        i = start + lt[QUICK_THREADS_IN_BLOCK] + threadIdx.x;
        for (; i < end - gt[QUICK_THREADS_IN_BLOCK]; i += QUICK_THREADS_IN_BLOCK) {
            out[i] = pivot;
        }
        if (threadIdx.x == 0) {
            int lt_sum = lt[QUICK_THREADS_IN_BLOCK];
            int gt_sum = gt[QUICK_THREADS_IN_BLOCK];
            long_seq = DevArray(start, start + lt_sum);
            short_seq = DevArray(end - gt_sum, end);
            if (lt_sum <= gt_sum) {
                swap(long_seq, short_seq);
            }
        }

        __syncthreads();
        altOrPush(long_seq, workStack, workStcIndex , out, tab);
        __syncthreads();
        altOrPush(short_seq, workStack,workStcIndex, out, tab);
        __syncthreads();

        //todo imporve it
        i = gstart + threadIdx.x;
        for (; i < gend; i += QUICK_THREADS_IN_BLOCK) {
            in[i] = out[i];
        }
    }
}

}