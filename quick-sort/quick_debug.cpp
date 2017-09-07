
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
        assert(L <= R);
        int d = wu.pivot;
        std::sort(toSort + L, toSort + d);
        std::sort(toSort + d + 1, toSort + R + 1);

        assert(toSort[d] == wu.pivot);

        for (int i = L; i <= R; ++i) {
            if (toSort[i] != i) {
                for (int j = i - 2; j < i + 20; ++j) {
                    PRINT1("%d %d\n", toSort[j], j);
                }
            }
            assert(toSort[i] == i);
        }
    }

    for (WorkUnit wu: done) {
        int L = wu.seq.start;
        int R = wu.seq.end - 1;
        assert(L <= R);
        int d = wu.pivot;
        std::sort(toSort + L, toSort + R + 1);
        if (toSort[d] != wu.pivot) {
            for (int i = -1; i < 2; ++i) {
                PRINT1("d p v i %d %d\n", toSort[d + i], d + i);
            }
        }
        assert(toSort[d] == wu.pivot);


        for (int i = L; i <= R; ++i) {
            if (toSort[i] != i) {
                for (int j = i - 2; j < i + 20; ++j) {
                    PRINT1("done v %d i %d\n", toSort[j], j);
                }
            }
            assert(toSort[i] == i);
        }
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
}
