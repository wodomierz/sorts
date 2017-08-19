//
// Created by Miron Ficak on 03/08/2017.
//

#include <cassert>
#include "PrefsumContext.h"
#include "sample_rand_context.h"
#include "sample_rand.h"

namespace sample_rand {

    Context::Context(int size) : baseData(size, BLOCK_SIZE) {
        sample_offsets = cuAllocHostInts(S_SIZE + 1);
        sample_offsets[0] = 0;
        bstPtr = cuAllocInts(S_SIZE);
        blockPrefsums = cuAllocInts(prefsumSize());
        cuMemsetD32(blockPrefsums, 0, prefsumSize());

        deviceToSort = cuAllocInts(size);
        out = cuAllocInts(size);
        cuMemsetD32(out, 0, size);
    }

    void Context::clean() {

        cuMemFree(deviceToSort);
        cuMemFree(out);
        cuMemFree(blockPrefsums);
        cuMemFree(bstPtr);
    }

    Context::Context(Context &memory, int sample_nr) : Context(memory) {
        deviceToSort = addIntOffset(deviceToSort, sample_offsets[sample_nr]);
        out = addIntOffset(out, sample_offsets[sample_nr]);
        int size = sample_offsets[sample_nr + 1] - sample_offsets[sample_nr];
        assert(size >= 0);
        baseData = BaseData(size, BLOCK_SIZE);
        //wydajniej
        if (baseData.size > M) {
            sample_offsets = cuAllocHostInts(S_SIZE + 1);
        }
        sample_offsets[0] = 0;

    }

    int Context::prefsumSize() {
        return baseData.number_of_blocks * S_SIZE;
    }

    void Context::localClean() {
        if (baseData.size > M) {
            cuMemFreeHost(sample_offsets);
        }

    }
}