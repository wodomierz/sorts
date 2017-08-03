//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_PREFSUM_CONTEXT_H
#define SORTS_PREFSUM_CONTEXT_H


#include "../utils/BaseData.h"

namespace sample_rand {
    class PrefsumContext {
    public:
        BaseData baseData;
        int *batchSums;

        void clean();

        PrefsumContext(int);
    };
}

#endif //SORTS_PREFSUM_CONTEXT_H
