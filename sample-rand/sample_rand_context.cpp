#include <cassert>
#include "PrefsumContext.h"
#include "sample_rand_context.h"
#include "sample_rand.h"

namespace sample_rand {

    Context::Context(int size) : baseData(size, BLOCK_SIZE), offset(0) {
        sample_offsets = cuAllocHostInts((S_SIZE + 1)*maxNumberOfBigWorkUnits());
        sample_offsets[0] = 0;
        bstPtr = cuAllocInts(S_SIZE*maxNumberOfBigWorkUnits());
        blockPrefsums = cuAllocInts(allPrefsumsCapacity());
        cuMemsetD32(blockPrefsums, 0, prefsumSize());

        deviceToSort = cuAllocInts(size);
        out = cuAllocInts(size);
        cuMemcpyDtoD(out, deviceToSort, size * sizeof(int)); //or inside kernel ??? it is ude because move result is in host
//        cuMemsetD32(out, 0, size);
    }

    void Context::clean() {

        cuMemFree(deviceToSort);
        cuMemFree(out);
        cuMemFree(blockPrefsums);
        cuMemFree(bstPtr);
    }

    Context::Context(Context &globalContext,int offset,int size, int prefsum_offset,int big_work_offset) : Context(globalContext) {
        this->offset = offset;
        this->baseData = BaseData(size, BLOCK_SIZE);


        deviceToSort = addIntOffset(deviceToSort, offset);
        out =addIntOffset(out, offset);
        bstPtr = addIntOffset(bstPtr, big_work_offset * S_SIZE);
        sample_offsets = sample_offsets + big_work_offset * (S_SIZE + 1);
        sample_offsets[0] = 0;
        blockPrefsums = addIntOffset(blockPrefsums, prefsum_offset);
    }


    int Context::prefsumSize() {
        return baseData.number_of_blocks * S_SIZE;
    }

//    void Context::localClean() {
//        if (baseData.size > SAMPLE_OTHER_SORT_LIM) {
//            cuMemFreeHost(sample_offsets);
//        }
//
//    }

    CUdeviceptr Context::relativeIn() {
        return addIntOffset(deviceToSort, offset);
    }

    CUdeviceptr Context::relativeOut() {
        return addIntOffset(out, offset);
    }
    template <int BlockSize, int M, int SampleSize>
    int maxNumberOfBigWorks(int size) {
        return size / (M+1);
    }
    int Context::maxNumberOfBigWorkUnits() {
        return maxNumberOfBigWorks<BLOCK_SIZE, SAMPLE_OTHER_SORT_LIM, S_SIZE>(baseData.size);
    }


    template <int BlockSize, int M, int SampleSize>
    int allPrefSumCapacityGen(int size) {
        return  maxNumberOfBigWorks<BlockSize, M, SampleSize>(size) * (2*M / BlockSize) * SampleSize;
    }
    int Context::allPrefsumsCapacity() {
        return allPrefSumCapacityGen<BLOCK_SIZE, SAMPLE_OTHER_SORT_LIM, S_SIZE>(baseData.size);
    }


}