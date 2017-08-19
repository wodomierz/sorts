//
// Created by Miron Ficak on 14/08/2017.
//

#include <cassert>
#include "quick_debug.h"
#include "../utils/utils.h"
#include <algorithm>

void checkMem() {
    DevArray *doneArrays1;
    cuMemAllocHost((void **) &doneArrays1, sizeof(DevArray) * 10);
    doneArrays1[0] = DevArray(1, 2);
    cuMemFreeHost(doneArrays1);
}


void
assertToSort(CUdeviceptr &out, CUdeviceptr &in, std::vector<WorkUnit> &work, std::vector<WorkUnit> &done, int size) {
    int *toSort = cuMemAllocH<int>(size);
    cuMemcpyDtoH(toSort, out, sizeof(int) * size);

    for (WorkUnit wu: work) {
        int L = wu.seq.start;
        int R = wu.seq.end - 1;
//            PRINT1("WORK: %d %d\n", L,R);
        assert(L <= R);
        int d = wu.pivot;
//        std::sort(toSort, toSort + size);
        std::sort(toSort + L, toSort + d);
        std::sort(toSort + d + 1, toSort + R + 1);

//            print_tab(toSort + L, d - L, d -L, "first half SORTED");

//            PRINT1("pivot %d %d %d\n", toSort[d],wu.pivot, d);
        assert(toSort[d] == wu.pivot);
//            print_tab(toSort + d + 1, R- d, R - d, "second half SORTED");

        for (int i = L; i <= R; ++i) {
            if (toSort[i] != i) {
                for (int j = i - 2; j < i + 20; ++j) {
                    PRINT1("%d %d\n", toSort[j], j);
                }
            }
            assert(toSort[i] == i);
        }
//            assert(toSort[d - 1] < toSort[d + 1]);
    }

    for (WorkUnit wu: done) {
        int L = wu.seq.start;
        int R = wu.seq.end - 1;
//            PRINT1("DONE: %d %d\n", L,R);
        assert(L <= R);
        int d = wu.pivot;
//        std::sort(toSort, toSort + size);
        std::sort(toSort + L, toSort + R + 1);
//            std::sort(toSort + d + 1, toSort + R +1);

//            print_tab(toSort + L, d - L, d -L, "first half SORTED");


        if (toSort[d] != wu.pivot) {
            for (int i = -1; i < 2; ++i) {
                PRINT1("d p v i %d %d\n", toSort[d + i], d + i);
            }
        }
        assert(toSort[d] == wu.pivot);

//            print_tab(toSort + d + 1, R- d, R - d, "second half SORTED");

        for (int i = L; i <= R; ++i) {
            if (toSort[i] != i) {
                for (int j = i - 2; j < i + 20; ++j) {
                    PRINT1("done v %d i %d\n", toSort[j], j);
                }
            }
            assert(toSort[i] == i);
        }
//            assert(toSort[d - 1] < toSort[d + 1]);
    }

    for (int i = 0; i < size; ++i) {
        if (toSort[i] != i) {
            print_Devtab(in, size, 10, i - 1, "IN");
            print_Devtab(out, size, 10, i - 1, "OUT");
            PRINT1("v %d i %d\n\n", toSort[i], i);
            for (int j = i - 1; j < i + 20; ++j) {
                PRINT1("v %d i %d\n", toSort[j], j);
            }

            assert(false);
        }

    }
    cuMemFreeHost(toSort);
//
}
