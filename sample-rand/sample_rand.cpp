//
// Created by Miron Ficak on 09/07/2017.
//
#include "sample_rand.h"

#include "cuda.h"
#include <cstdio>
#include <iostream>
#include <ctime>
#include "../utils/utils.h"
#include "sample_rand_context.h"
#include "sample_rand_device.h"
#include <algorithm>
#include <assert.h>

using namespace std;


void create_search_tree(sample_rand::Context &memory) {
    //WORKS ONLY IF memory.size > S_SIZE
    int *tree = cuAllocHostInts(S_SIZE);
    int *sample = cuAllocHostInts(S_SIZE);
    int *to_sort = cuAllocHostInts(memory.baseData.size);
    cuMemcpyDtoH(to_sort, memory.deviceToSort, memory.baseData.size * sizeof(int));
    int delta = memory.baseData.size / S_SIZE;

    std::sort(to_sort, to_sort + memory.baseData.size);

    for (int i = (delta - 1); i < memory.baseData.size; i += delta) {
        sample[i / delta] = to_sort[i];
    }

    std::sort(sample, sample + S_SIZE);
    int iteratr = 0;
    for (int i = 2; i <= S_SIZE; i *= 2) {
        for (int j = 1; j < i; j += 2) {
            tree[iteratr++] = sample[j * (S_SIZE / i) - 1];
        }
    }
    tree[S_SIZE - 1] = 0;
    cuMemcpyHtoD(memory.bstPtr, tree, S_SIZE * sizeof(int));
    cuMemFreeHost(tree);
    cuMemFreeHost(sample);
}

inline void prefsum(sample_rand::Context &memory, sample_rand::Device &device) {
    sample_rand::PrefsumContext prefsumMemory(memory.prefsumSize());
    device.localPrefSums(memory, prefsumMemory);

    device.prefsumOfBatchSums(prefsumMemory);


    device.globalPrefSums(memory, prefsumMemory);
    prefsumMemory.clean();
}

//int counter = 0;

void sampleRand(sample_rand::Device &device, sample_rand::Context &memory) {
    create_search_tree(memory);

    device.counters(memory);

//    print_Devtab(memory.blockPrefsums, memory.prefsumSize(),memory.prefsumSize(), 0,"BEF");
    prefsum(memory, device);
//    print_Devtab(memory.blockPrefsums, memory.prefsumSize(),memory.prefsumSize(), 0,"aft");


    device.scatter(memory);
    memory.moveResult();

    // could be more efficient
//    PRINT1("father %seq\n", memory.baseData.size);
    for (int i = 0; i < S_SIZE; ++i) {
        int offset = memory.sample_offsets[i];
//        PRINT1( "o %seq f %seq\n", offset, memory.baseData.size);
        int size = memory.sample_offsets[i + 1] - memory.sample_offsets[i];
//        PRINT1( "o1 %seq f %seq\n", memory.sample_offsets[i + 1], memory.baseData.size);
        if (size > 0) {
            sample_rand::Context mem(memory, i);
            assertPrintable(([i, memory, mem] {
                PRINT1("%seq %seq %seq %seq %seq\n",
                       i,
                       memory.sample_offsets[i],
                       memory.sample_offsets[i + 1],
                       memory.baseData.size,
                       mem.baseData.size);
            }),
                            memory.baseData.size != mem.baseData.size);
            //could be more efficient
            if (mem.baseData.size > M) {
                sampleRand(device, mem);
            } else {
                device.chujowy(mem);
            }
            if (memory.baseData.size > 5000 && mem.baseData.size == 1) {
                print_Devtab(
                        memory.blockPrefsums,
                        memory.prefsumSize(),
                        memory.prefsumSize(),
                        1,
                        "PREFSUM JUMPS",
                        indexedPrint,
                        [](int i, int *tab) -> bool { return (tab[i] > tab[i - 1]); }
                );
                print_Devtab(
                        memory.bstPtr,
                        S_SIZE,
                        S_SIZE,
                        1,
                        "BST",
                        indexedPrint
                );
                print_Devtab(memory.deviceToSort, memory.baseData.size, 512);
                PRINT1("\nWAT? %seq %seq %seq %seq\n", i, memory.baseData.size, memory.sample_offsets[i], memory.sample_offsets[i + 1]);
                assert(false);
            }
        }

    }
    memory.localClean();
}


void sampleRand(int *to_sort, int size) {
    int cudaVersion;
    cuDriverGetVersion(&cudaVersion);
    PRINT1("running version %seq\n", cudaVersion);

    sample_rand::Device device;
    sample_rand::Context memory(size);

    cuMemHostRegister((void *) to_sort, size * sizeof(int), 0);
    cuMemcpyHtoD(memory.deviceToSort, to_sort, size * sizeof(int));

    PRINT1("beforedsa %seq\n", size);
    sampleRand(device, memory);
    PRINT("after\n");

    cuMemcpyDtoH((void *) to_sort, memory.deviceToSort, size * sizeof(int));

    memory.clean();
    cuMemHostUnregister(to_sort);
    cuCtxDestroy(device.cuContext);

}

