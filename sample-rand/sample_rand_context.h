//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_SAMPLE_RAND_CONTEXT_H
#define SORTS_SAMPLE_RAND_CONTEXT_H


#include <cuda.h>
#include "../utils/BaseData.h"

namespace sample_rand {
    class Context {
    public:
        BaseData baseData;
        int offset;
        int *sample_offsets;
        CUdeviceptr prefsumsMem;

        CUdeviceptr blockPrefsums;
        CUdeviceptr deviceToSort;
        CUdeviceptr out;
        CUdeviceptr bstPtr;


        int allPrefsumsCapacity();
        int maxNumberOfBigWorkUnits();
        int prefsumSize();

        void moveResult() {
            cuMemcpy(deviceToSort, out, sizeof(int) * baseData.size);
//            cuMemsetD32(out, 0, baseData.size);
        }

//        void localClean();

        void clean();

        Context(int size);
        Context(Context &globalContext,int offset,int prefsum_offset,int prefMemOff, int size, int big_work_offset);

//        Context(Context &memory, int);

        CUdeviceptr relativeIn();
        CUdeviceptr relativeOut();
    };

}
#endif //SORTS_SAMPLE_RAND_CONTEXT_H
