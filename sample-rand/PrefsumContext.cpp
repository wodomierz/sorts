//
// Created by Miron Ficak on 03/08/2017.
//

#include "PrefsumContext.h"
#include "sample_rand.h"
#include "../utils/BaseData.h"

namespace sample_rand {

    PrefsumContext::PrefsumContext(int size, CUdeviceptr batchSums) : baseData(size, PREFSUM_BLOCK_SIZE), batchSums(batchSums) {
    }

}
