#include <vector>
#include "quick_sort.h"
#include "cuda.h"
#include "../utils/utils.h"
#include "quick_sort_device.h"
#include "quick_shared.h"
#include <list>
#include <algorithm>
#include <cassert>
//PIVOTY CO ZE MNA ROBICIE
int pivot(CUdeviceptr to_sort, int size) {
    //TODO
    int *copy = cuMemAllocH<int>(size);
    cuMemcpyDtoH(copy, to_sort, size * sizeof(int));
//    PRINT1("co do chuja %d %d %d", copy[0], copy[1], copy[2]);
    std::sort(copy, copy + size);
    return copy[(size -1) / 2];
}

int sum_seq_size(std::vector<WorkUnit> work, int max_seq) {
    int result = 0;
    for (WorkUnit unit : work) {
        result += (unit.seq.end - unit.seq.start)/max_seq;
    }
    return result;
}

void checkMem(){
    DevArray *doneArrays1;
    cuMemAllocHost((void**)&doneArrays1, sizeof(DevArray)*10);
    doneArrays1[0] =DevArray(1,2);
    cuMemFreeHost(doneArrays1);
}


void prepareBlocks(Block* blocks, SharedVars* parents, std::vector<WorkUnit>& work, int block_size ){
    int total_block_index = 0;
    int seq_index = -1;

    for (WorkUnit unit : work) {
        int block_count = ceil_div(unit.seq.end - unit.seq.start, block_size);
        parents[++seq_index] = SharedVars(
            unit.seq,
            unit.seq,
            block_count,
            seq_index
        );
        int i;
        for (i = 0; i < block_count - 1; i++) {
            int bstart = unit.seq.start + block_size * i;
            blocks[total_block_index++] = {
                {
                    //???
                    {bstart, bstart + block_size},
                    unit.pivot
                },
                parents + seq_index
            };
            DevArray &array = blocks[total_block_index - 1].workUnit.seq;
//                PRINT1("add block %d %d %d\n", array.start, array.end, total_block_index -1);
        }
        blocks[total_block_index++] = {
            {
                {
                    unit.seq.start + block_size * i,
                    unit.seq.end
                },
                unit.pivot
            },
            parents +seq_index
        };
        DevArray &array = blocks[total_block_index - 1].workUnit.seq;
//            PRINT1("add block %d %d\n", array.start, array.end);


    }
}

void sort(int size, CUdeviceptr &in, CUdeviceptr &out) {
    quick::Device device;

    int start_pivot = pivot(in, size);
    std::vector<WorkUnit> work = {WorkUnit(DevArray(0, size), start_pivot)};
    std::vector<WorkUnit> done;

    int max_seq = MAX_SEQ;

//    print_Devtab(in, size, size,0, "IN");

//    int L =0, R=size-1;
    int block_size = ceil_div(size, max_seq); //WTF??? block_size w innym miejscy niż w kodzie, no ale do kurwy nędzy ten max_seq to jakaś żenada
//        sum_seq_size(work, max_seq);
    while (!work.empty() && work.size() + done.size() < max_seq) {

//        PRINT1("block size %d %d\n", block_size,  size / max_seq);
//
//        PRINT1("while %d %d %d\n", block_size, sum_seq_size(work), max_seq);

        int total_block_count = 0;
        for (WorkUnit unit : work) {
            assert(arraySize(unit.seq) > block_size);
            int block_count = ceil_div(unit.seq.end - unit.seq.start, block_size);
            total_block_count += block_count;
        }

        //consider vector and register array of this vector??
        Block *blocks = cuMemAllocH<Block>(total_block_count);


        SharedVars* parents = cuMemAllocH<SharedVars>(work.size());

        prepareBlocks(blocks, parents, work, block_size);

        int seq_num = work.size();
//        PRINT1("alloc ? gqsort %d\n", seq_num);
        //TODO
        WorkUnit *news = cuMemAllocH<WorkUnit>(2*work.size());

//        checkMem();
//        checkMem();
//        PRINT1("before gqsort\n");
//        print_Devtab(out, size, size,0, "BEF GQ");
        device.gqsort(blocks, total_block_count, in, out, news);

//        checkMem();
//        checkMem();
//        PRINT1("afetr gqsort %d %d %d\n", blocks[0].sharedVars->seq.start, blocks[1].sharedVars->seq.end, blocks[0].sharedVars->block_count);
//        print_Devtab(in, size, size,0, "IN");
//        print_Devtab(out, size, size,0, "GQSORT");
        int* toSort = cuMemAllocH<int>(size);
        cuMemcpyDtoH(toSort, out, sizeof(int) * size);

        for (WorkUnit wu: work) {
            int L = wu.seq.start;
            int R = wu.seq.end -1;
            PRINT1("WORK: %d %d\n", L,R);
            assert(L <= R);
            int d = wu.pivot;
//        std::sort(toSort, toSort + size);
            std::sort(toSort + L, toSort + d);
            std::sort(toSort + d + 1, toSort + R +1);

//            print_tab(toSort + L, d - L, d -L, "first half SORTED");

            PRINT1("pivot %d %d %d\n", toSort[d],wu.pivot, d);
            assert(toSort[d] == wu.pivot);
//            print_tab(toSort + d + 1, R- d, R - d, "second half SORTED");

            for (int i=L; i <= R; ++i) {
                if (toSort[i] != i ) {
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
            int R = wu.seq.end -1;
            PRINT1("DONE: %d %d\n", L,R);
            assert(L <= R);
            int d = wu.pivot;
//        std::sort(toSort, toSort + size);
            std::sort(toSort + L, toSort + R +1);
//            std::sort(toSort + d + 1, toSort + R +1);

//            print_tab(toSort + L, d - L, d -L, "first half SORTED");


            if (toSort[d] != wu.pivot) {
                for (int i=-1; i<2; ++i) {
                    PRINT1("d p v i %d %d\n", toSort[d + i], d+ i);
                }
            }
            assert(toSort[d] == wu.pivot);

//            print_tab(toSort + d + 1, R- d, R - d, "second half SORTED");

            for (int i=L; i <= R; ++i) {
                if (toSort[i] != i ) {
                    for (int j = i - 2; j < i + 20; ++j) {
                        PRINT1("done v %d i %d\n", toSort[j], j);
                    }
                }
                assert(toSort[i] == i);
            }
//            assert(toSort[d - 1] < toSort[d + 1]);
        }

        for (int i=0; i< size; ++i) {
            if (toSort[i] != i) {
                print_Devtab(in, size, 10, i-1, "IN");
                print_Devtab(out, size, 10, i-1, "OUT");
                PRINT1("v %d i %d\n\n", toSort[i], i);
                for (int j = i - 1; j < i + 20; ++j) {
                    PRINT1("v %d i %d\n", toSort[j], j);
                }

                assert(false);
            }

        }
        cuMemFreeHost(toSort);
//        PRINT1("%d %d %d %d", news[0].pivot, news[0])
//        checkMem();
        //???
//        PRINT1("work clear?\n");
//        assert(false);
        work.clear();
//        checkMem();
//        PRINT1("work cleared\n");
        for (int i = 0; i < seq_num; ++i) {
            WorkUnit &workUnitL = news[2*i];
            WorkUnit &workUnitR = news[2*i+1];
//            checkMem();
//            PRINT1("work unit get %d %d %d\n", i, workUnitL.seq.start, workUnitL.seq.end);

            if (arraySize(workUnitL.seq) <= block_size) { //diff algo
//                if (arraySize(workUnitL.seq) > 1) {
                    done.push_back(workUnitL);
//                }
            } else {
                work.push_back(workUnitL);
            }

            if (arraySize(workUnitR.seq) <= block_size) { //diff algo
//                if (arraySize(workUnitR.seq) > 1) {
                    done.push_back(workUnitR);
//                }
            } else {

                work.push_back(workUnitR);
            }
        }
//        PRINT1("to work %d", done.size() + work.size());
//        print_Devtab(in, size, 30,0, "IN");
//        print_Devtab(out, size, 30,0, "OUT");

//        std::swap(in, out);

        //todo improve
        cuMemcpyDtoD(in, out, size* sizeof(int));

//        checkMem();
        cuMemFreeHost(blocks);
//        PRINT1("free blocks\n");
        cuMemFreeHost(news);
        cuMemFreeHost(parents);
//        PRINT1("free news\n");
//        checkMem();
    }

    done.insert(done.end(), work.begin(), work.end());
//    PRINT1("insert\n");

    //improve adding to done
//    checkMem();


//    int* toSort = cuMemAllocH<int>(size);
//    cuMemcpyDtoH(toSort, in, sizeof(int) * size);



    DevArray* doneArrays = cuMemAllocH<DevArray>(done.size());
//    PRINT1("alloc done arrays\n");
    for (int i = 0; i < done.size(); i++) {
//        PRINT1("bef copied: st %d end %d piv %d\n", done[i].seq.start,  done[i].seq.end, toSort[done[i].seq.end]);
        doneArrays[i] = done[i].seq;
//        PRINT1("array copied: %d\n", i);
    }


    print_Devtab(in, size, 10, 0,"BEF RESULT", indexedPrint);
    print_Devtab(out, size, 10, 0,"BEF RESULT OUT", indexedPrint);
//    checkMem();
//    checkMem();
//    checkMem();
//    checkMem();
//    PRINT1("lq bef\n");

    //TODO fix it
    cuMemcpyDtoD(out, in, sizeof(int) *size);
    device.lqsort(doneArrays, done.size(), in, out);
    cuMemcpyDtoD(in, out, sizeof(int) *size);
    checkMem();
    checkMem();
    checkMem();
    checkMem();
    checkMem();
    checkMem();
    checkMem();
    checkMem();


//    PRINT1("lq after\n");
    print_Devtab(out, size, 10, 0,"RESULT", indexedPrint);
    cuMemFreeHost(doneArrays);
//    PRINT1("free arrays");
}


void quick_sort_device(CUdeviceptr to_sort, int size) {
    CUdeviceptr in = to_sort;
    CUdeviceptr out = cuAllocInts(size);
    cuMemsetD32(out, 0, size);
    sort(size, in, out);

//    PRINT1("\nCODOCHUJA %lu %lu %lu\n", to_sort, in, out);

    if (out != to_sort) {
        cuMemcpyDtoD(to_sort, out, sizeof(int) * size);
        cuMemFree(out);
    } else {
        cuMemFree(in);
    }
}

void quick_sort(int *to_sort, int size) {
    cuInit(0);
    CUdevice cuDevice;
    CUcontext cuContext;
    manageResult(cuDeviceGet(&cuDevice, 0), "cannot acquire device");
    manageResult(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");

    PRINT1("tutej\n");
    cuMemHostRegister(to_sort, sizeof(int) * size, 0);
    CUdeviceptr in = cuAllocD<int>(size);
    cuMemcpyHtoD(in, to_sort, size* sizeof(int));
    quick_sort_device(in, size);
    cuMemcpyDtoH(to_sort, in, sizeof(int) * size);
    cuMemFree(in);

    cuCtxDestroy(cuContext);
}