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
        int *sample_offsets;
        CUdeviceptr blockPrefsums;
        CUdeviceptr deviceToSort;
        CUdeviceptr out;
        CUdeviceptr bstPtr;


        int prefsumSize();

        void moveResult() {
            cuMemcpy(deviceToSort, out, sizeof(int) * baseData.size);
            cuMemsetD32(out, 0, baseData.size);
        }

        void localClean();

        void clean();

        Context(int);

        Context(Context &memory, int);
    };

}
#endif //SORTS_SAMPLE_RAND_CONTEXT_H
