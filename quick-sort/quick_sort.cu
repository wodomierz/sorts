
#include "quick_sort.h"
#include "../utils/cuda_device.h"
#include <thrust/device_vector.h>

extern "C" {


//__global__ int s_start;
//__global__ int s_end;
//__global__ int old_start;
//__global__ int old_end;
//__global__ int block_count;


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
void alt_sort(DevArray array, int *in, int *out) {
    for (int i =1; i < array.size; ++i) {
        for (int j=0; j< i; ++j) {
            min_max(in, array.start + j, array.start + i, array.end);
        }
    }
    for (int i=array.start; i< array.end; ++i) {
        out[i] = in[i];
    }
}

__device__
int median(int *array, int i1, int i2, int i3) {
    int tab[3] = {array[i1], array[i2], array[i3]};
    //improve
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) {
            int max = tab[i];
            int min = tab[j];
            if (tab[i] < tab[j]) {
                int key = tab[i];
                tab[i] = tab[j];
                tab[j] = key;
            }
        }
    }
    return tab[1];
}

__device__
void pref_sum(int *array) {
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
void gqsort(Block *blocks, int *in, int *out, WorkUnit *news) {
    //cached in to shared ?
    __shared__ int lt[QUICK_THREADS_IN_BLOCK],
        gt[QUICK_THREADS_IN_BLOCK],
        pivot,
        start,
        end,
        lbeg, gbeg;
//    __shared__ int* s;
//    __shared__ int* out_s;
    int i, l_from, g_from, l_pivot, g_pivot;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int gthid = threadIdx.x + blockId * blockDim.x;

    SharedVars *parent;

    if (threadIdx.x == 0) {
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

    for (; i < end; i += QUICK_THREADS_IN_BLOCK) {
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
    for (; i < end; i += QUICK_THREADS_IN_BLOCK) {
        if (in[i] < pivot) {
            out[l_from++] = in[i];
        }
        if (in[i] > pivot) {
            out[g_from++] = in[i];
        }
    }
    if (threadIdx.x == 0 && atomicSub(&parent->block_count, 1)) {
        for (i = parent->seq1.start; i < parent->seq1.end; i++) {
            in[i] = pivot;
        }
        int old_start = parent->seq2.start;
        int old_end = parent->seq2.end - 1;
        int s_start = parent->seq1.start;
        int s_end = parent->seq1.end - 1;
        l_pivot = median(out, old_start, (old_start + s_start) / 2, s_start);
        g_pivot = median(out, s_end, (s_end + old_end) / 2, old_end);

        news[parent->seq_index] = WorkUnit(DevArray(old_start, s_start), l_pivot);
        news[parent->seq_index] = WorkUnit(DevArray(s_end, old_end), g_pivot);
    }
}

#define LAST_THREAD (QUICK_THREADS_IN_BLOCK - 1)

__device__ __forceinline__
void altOrPush(
    DevArray &devArray,
    thrust::device_vector<DevArray> &work_stack,
    int *in,
    int *out) {
    if (devArray.size <= OTHER_SORT_LIM) {
        alt_sort(devArray, in, out);
    } else work_stack.push_back(devArray);
}

__global__
void lqsort(DevArray *seqs, int **in_h, int **out_h) {
    __shared__
    int lt[QUICK_THREADS_IN_BLOCK],
        gt[QUICK_THREADS_IN_BLOCK],
        pivot;
    // how with shared memory ???
    __shared__ DevArray newseq1(0, 0);
    __shared__ DevArray newseq2(0, 0);
    __shared__ DevArray s(0, 0);
//        s[QUICK_THREADS_IN_BLOCK];
    //???
    //maybe pointer???
    thrust::device_vector<DevArray> work_stack;
    int* out = *out_h;
    int* in = *in_h;
    int i, l, l_from, g_from;

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;

    if (threadIdx.x == 0) {
        work_stack.push_back(seqs[blockId]);
    }
    while (!work_stack.empty()) {
        //WTF???
        s = *(&work_stack.back()).get();
        work_stack.pop_back();
        // how to work_stack.back().value_type();

        pivot = median(in, s.start, (s.end - 1 + s.start) / 2, s.end - 1);
        lt[threadIdx.x] = 0;
        gt[threadIdx.x] = 0;
        //synch?
        i = s.start + threadIdx.x;
        for (; i < s.end; i += QUICK_THREADS_IN_BLOCK) {
            if (in[i] < pivot) {
                lt[threadIdx.x]++;
            }
            if (in[i] > pivot) {
                gt[threadIdx.x]++;
            }
        }
        pref_sum(lt);
        pref_sum(gt);
        l_from = s.start + lt[threadIdx.x];
        g_from = s.end - gt[threadIdx.x];

        i = s.start + threadIdx.x;
        for (; i < s.end; i += QUICK_THREADS_IN_BLOCK) {
            if (in[i] < pivot) {
                out[l_from++] = in[i];
            }
            if (in[i] > pivot) {
                out[g_from++] = in[i];
            }
        }

        if (threadIdx.x == LAST_THREAD) {
            out[s.start + lt[LAST_THREAD]] = pivot;

            int lt_sum = lt[LAST_THREAD];
            int gt_sum = gt[LAST_THREAD];
            DevArray long_seq(s.start, s.start + lt_sum);
            DevArray short_seq(s.end - gt_sum, s.end);
            if (lt_sum < gt_sum) {
                //?
                std::swap(long_seq, short_seq);
            }
            altOrPush(long_seq, work_stack, in, out);
            altOrPush(short_seq, work_stack, in, out);
        }
        std::swap(in, out);
    }
    *out_h = in;
    *in_h = out;
}

}