
#include "quick_sort.h"
#include "../utils/cuda_device.h"
#include "quick_shared.h"
#include <thrust/device_vector.h>


template <typename T>
__device__ __forceinline__
void swap(T& a, T&b) {
    T c  = a;
    a = b;
    b = c;
}

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

__device__
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
    __syncthreads();
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
    int *out) {
    if (devArray.end - devArray.start <= OTHER_SORT_LIM) {
        alt_sort(devArray, out);
    } else work_stack[++stackIndex] = devArray;
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
//        s[QUICK_THREADS_IN_BLOCK];
    //???
    //maybe pointer???
//    thrust::device_vector<DevArray> work_stack;
    int debind = 0;
    int *out = out_h;
    int *in = in_h;
    int i, l_from, g_from;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    if (threadIdx.x == 0) {
        workStcIndex = 0;
        workStack[0] = seqs[blockId];
        gstart = workStack[0].start;
        gend = workStack[0].end;

//        work_stack.push_back(seqs[blockId]);
    }
    __syncthreads();
    while (workStcIndex >= 0) {
        //WTF???
        if (threadIdx.x == 0) {
            DevArray dev = workStack[workStcIndex--];
            start = dev.start;
            end = dev.end;
            lt[threadIdx.x + QUICK_THREADS_IN_BLOCK] = 0;
            gt[threadIdx.x + QUICK_THREADS_IN_BLOCK] = 0;
        }
        __syncthreads();
        // how to work_stack.back().value_type();

        if(threadIdx.x ==0) {
            pivot = median(in, start, end);
        }
        lt[threadIdx.x] = 0;
        gt[threadIdx.x] = 0;
        __syncthreads();
        //synch?
        i = start + threadIdx.x;
        for (; i < end; i += QUICK_THREADS_IN_BLOCK) {
            if (in[i] < pivot) {
                lt[threadIdx.x]++;
            }
            if (in[i] > pivot) {
                gt[threadIdx.x]++;
            }
        }
        __syncthreads();
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
        __syncthreads();
        i = start + lt[QUICK_THREADS_IN_BLOCK] + threadIdx.x;
        for (; i < end - gt[QUICK_THREADS_IN_BLOCK]; i += QUICK_THREADS_IN_BLOCK) {
            out[i] = pivot;
            in[i] = pivot;
        }
        __syncthreads();
        if (threadIdx.x == 0) {

            int lt_sum = lt[QUICK_THREADS_IN_BLOCK];
            int gt_sum = gt[QUICK_THREADS_IN_BLOCK];
            DevArray long_seq(start, start + lt_sum);
            DevArray short_seq(end - gt_sum, end);
            if (lt_sum <= gt_sum) {
                //?
                swap(long_seq, short_seq);
            }
//            if (blockId == 0){
//                debugger[debind++] = gstart;
//                debugger[debind++] = gend;
//                debugger[debind++] = -8000;
//                debugger[debind++] = start;
//                debugger[debind++] = pivot;
//                debugger[debind++] = end;
//                debugger[debind++] = lt_sum;
//                debugger[debind++] = gt_sum;
//                debugger[debind++] = -3000;
//            }

//            if (blockId == 0 && short_seq.end - short_seq.start <= OTHER_SORT_LIM) {
//                debugger[debind++] = -2000;
//                for (int i= short_seq.start; i < short_seq.end;++i) {
//                    debugger[debind++] = out[i];
//                }
//                debugger[debind++] = -2000;
//            }
            altOrPush(long_seq, workStack, workStcIndex , out);
            altOrPush(short_seq, workStack,workStcIndex, out);

//            if (blockId == 0 && short_seq.end - short_seq.start <= OTHER_SORT_LIM) {
//                debugger[debind++] = -500;
//                debugger[debind++] = short_seq.start;
//                debugger[debind++] = short_seq.end;
//                debugger[debind++] = -500;
//                for (int i= short_seq.start; i < short_seq.end;++i) {
//                    debugger[debind++] = out[i];
//                }
//                debugger[debind++] = -500;
//            }
//            if (blockId == 0 && long_seq.end - long_seq.start <= OTHER_SORT_LIM) {
//                debugger[debind++] = -1000;
//                debugger[debind++] = long_seq.start;
//                debugger[debind++] = long_seq.end;
//                debugger[debind++] = -1000;
//
//                for (int i= long_seq.start; i < long_seq.end;++i) {
//                    debugger[debind++] = out[i];
//                }
//                debugger[debind++] = -1000;
//            }
        }
        __syncthreads();
        swap(in, out);
        __syncthreads();
    }
    __syncthreads();
    i = gstart + threadIdx.x;
    for (; i < gend; i += QUICK_THREADS_IN_BLOCK) {
        out_h[i] = i;
//            out[i];
//        in_h[i] = -20;
//            in[i];
    }
}

}