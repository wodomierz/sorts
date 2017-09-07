#ifndef SORTS_CUDE_DEVICE_H
#define SORTS_CUDE_DEVICE_H

#ifdef __JETBRAINS_IDE__

#include <host_defines.h>
#include <cuda_builtin_vars.h>

#define atomicExch
#define __syncthreads
#define atomicAdd
#define atomicSub
#endif

#endif //SORTS_CUDE_DEVICE_H
