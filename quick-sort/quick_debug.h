
#ifndef SORTS_QUICK_DEBUG_H

#include <cuda.h>
#include <vector>
#include "quick_shared.h"

void checkMem();

void
assertToSort(CUdeviceptr &out, CUdeviceptr &in, std::vector<WorkUnit> &work, std::vector<WorkUnit> &done, int size);

#define SORTS_QUICK_DEBUG_H

#endif //SORTS_QUICK_DEBUG_H
