//
// Created by Miron Ficak on 03/08/2017.
//

#include "PrefsumContext.h"
#include "sample_rand.h"


namespace sample_rand {

    PrefsumContext::PrefsumContext(int size) : baseData(size, PREFSUM_BLOCK_SIZE) {
        batchSums = cuAllocHostInts(baseData.number_of_blocks + 1);
    }

    void PrefsumContext::clean() {
        cuMemFreeHost(batchSums);
    }
}
