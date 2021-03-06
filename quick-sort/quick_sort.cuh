#ifndef SORTS_QUICK_SORT_CUH
#define SORTS_QUICK_SORT_CUH

#include "../utils/cuda_device.h"
#include "../prefsum/prefsum.cuh"
#include "quick_shared.h"
#include "../utils/kernel_commons.cuh"
#include "../bitonic/bitonic_sort.cuh"

__device__ __forceinline__
int median(int *array, int start, int end) {
    int tab[3] = {array[start], array[(start + end - 1) / 2], array[end - 1]};
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

template<int Threads>
__device__ __forceinline__
void pref_sum(int (&shared)[2][Threads], int *array) {
    shared[0][threadIdx.x] = array[threadIdx.x];
    __syncthreads();
    bool to = 0;
    prefixSumDev<Threads, 1>(shared, to);

    array[threadIdx.x + 1] = shared[to][threadIdx.x];
    if (threadIdx.x == 0) {
        array[0] = 0;
    }
}

template<int ThreadsPow>
__device__ __forceinline__
void gqsort_dev(Block *blocks, int *in, int *out, WorkUnit *news) {
    const int Threads = (1 << ThreadsPow);
    __shared__ int lt[Threads + 1],
        gt[Threads + 1],
        pivot,
        start,
        end,
        lbeg, gbeg;
    __shared__ int shared[2][Threads];


    int i, l_from, g_from, l_pivot, g_pivot;

    SharedVars *parent;

    if (threadIdx.x == Threads - 1) {
        Block block = blocks[one_dimension_blockId()];
        parent = block.sharedVars;
        start = block.workUnit.seq.start;
        end = block.workUnit.seq.end;
        pivot = block.workUnit.pivot;

        lt[threadIdx.x + 1] = 0;
        gt[threadIdx.x + 1] = 0;
    }


    lt[threadIdx.x] = 0;
    gt[threadIdx.x] = 0;
    __syncthreads();
    i = start + threadIdx.x;

    for (; i < end; i += Threads) {
        if (in[i] < pivot) lt[threadIdx.x]++;
        if (in[i] > pivot) gt[threadIdx.x]++;
    }
    pref_sum(shared, lt);
    pref_sum(shared, gt);


    if (threadIdx.x == Threads - 1) {
        lbeg = atomicAdd(&(parent->seq.start), lt[Threads]);
        gbeg = atomicSub(&(parent->seq.end), gt[Threads]) - gt[Threads];
    }
    __syncthreads();

    l_from = lbeg + lt[threadIdx.x];
    g_from = gbeg + gt[threadIdx.x];
    i = start + threadIdx.x;
    for (; i < end; i += Threads) {
        if (in[i] < pivot) {
            out[l_from++] = in[i];
        }
        if (in[i] > pivot) {
            out[g_from++] = in[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == Threads - 1) {
        lbeg = atomicSub(&(parent->block_count), 1);
        if (lbeg == 1) {
            start = parent->seq.start;
            end = parent->seq.end;
            l_pivot = median(out, parent->old_seq.start, parent->seq.start);
            g_pivot = median(out, parent->seq.end, parent->old_seq.end);
            news[2 * parent->seq_index] = WorkUnit(DevArray(parent->old_seq.start, parent->seq.start), l_pivot);
            news[2 * parent->seq_index + 1] = WorkUnit(DevArray(parent->seq.end, parent->old_seq.end), g_pivot);
        }
    }
    __syncthreads();
    if (lbeg == 1) {
        for (i = start + threadIdx.x; i < end; i += Threads) {
            out[i] = pivot;
        }
    }

}

template<int OtherSortPow>
__device__ __forceinline__
void altOrPush(
    DevArray &devArray,
    DevArray *work_stack,
    int &stackIndex,
    int *out,
    int *tab) {
    static int OtherSortLimit = (1 << OtherSortPow);
    if (devArray.end - devArray.start <= OtherSortLimit) {
        if (arraySize(devArray) > 1) {
            bitonic_merge_device<OtherSortPow - 1>(out + devArray.start, arraySize(devArray), tab);
        }
    } else {
        if (threadIdx.x == 0) work_stack[++stackIndex] = devArray;
    }
}

template<int ThreadsPow>
__device__ __forceinline__
void lqsort_dev(DevArray *seqs, int *in_h, int *out_h) {
    const int Threads = (1 << ThreadsPow);

    __shared__ int lt[Threads + 1];
    __shared__ int gt[Threads + 1];
    __shared__ int pivot, start, end;
    __shared__ int workStcIndex;
    __shared__ DevArray workStack[64];
    __shared__ int shared[2 * Threads];
    __shared__ DevArray long_seq, short_seq;

    int *out = out_h;
    int *in = in_h;
    int i, l_from, g_from;

    if (threadIdx.x == 0) {
        workStcIndex = 0;
        workStack[0] = seqs[one_dimension_blockId()];
    }
    __syncthreads();
    while (workStcIndex >= 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            DevArray dev = workStack[workStcIndex--];
            start = dev.start;
            end = dev.end;
            lt[Threads] = 0;
            gt[Threads] = 0;
            pivot = median(in, start, end);
        }
        lt[threadIdx.x] = 0;
        gt[threadIdx.x] = 0;
        __syncthreads();
        i = start + threadIdx.x;
        for (; i < end; i += Threads) {
            if (in[i] < pivot) {
                lt[threadIdx.x]++;
            }
            if (in[i] > pivot) {
                gt[threadIdx.x]++;
            }
        }

        pref_sum(*reinterpret_cast<int (*)[2][Threads]>(shared), lt); //exclusive
        __syncthreads();
        pref_sum(*reinterpret_cast<int (*)[2][Threads]>(shared), gt);
        __syncthreads();
        l_from = start + lt[threadIdx.x];
        g_from = end - gt[threadIdx.x + 1];
        i = start + threadIdx.x;
        for (; i < end; i += Threads) {
            if (in[i] < pivot) {
                out[l_from++] = in[i];
            }
            if (in[i] > pivot) {
                out[g_from++] = in[i];
            }
        }
        i = start + lt[Threads] + threadIdx.x;
        for (; i < end - gt[Threads]; i += Threads) {
            out[i] = pivot;
        }
        if (threadIdx.x == 0) {
            int lt_sum = lt[Threads];
            int gt_sum = gt[Threads];
            long_seq = DevArray(start, start + lt_sum);
            short_seq = DevArray(end - gt_sum, end);
            if (lt_sum <= gt_sum) {
                swap(long_seq, short_seq);
            }
        }

        __syncthreads();
        altOrPush<ThreadsPow + 1>(long_seq, workStack, workStcIndex, out, shared);
        __syncthreads();
        altOrPush<ThreadsPow + 1>(short_seq, workStack, workStcIndex, out, shared);
        __syncthreads();

        i = start + threadIdx.x;
        for (; i < end; i += Threads) {
            in[i] = out[i];
        }
    }
}

#endif //SORTS_QUICK_SORT_CUH