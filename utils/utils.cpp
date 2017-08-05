//
// Created by Miron Ficak on 28/06/2017.
//

#include <cassert>
#include "utils.h"
#include <algorithm>

void manageError(CUresult res, std::string message, int line, const char* file) {
    if (res != CUDA_SUCCESS) {
        printf("%s , error nr: %d file : %s line: %d\n", message.c_str(), res, file, line);
        exit(1);
    }
}


void print_tab(int* tab, int size, int prints, const char* title, Printer printer, Filter filter) {
    PRINT1("\n");
    PRINT1("%s\n", title);
    for (int i=0; i  < size && i < prints; ++i) {
        if (filter(i, tab)) {
            printer(i, tab);
        }
    }
    PRINT1("\n");
}

void print_Devtab(CUdeviceptr& dtab, int size, int prints, int from, const char* title, Printer printer, Filter filter) {
    int * tab;

    prints = std::min(size, prints);
    cuCtxSynchronize();
    cuMemAllocHost((void**) &tab, sizeof(int) * prints);

    cuMemcpyDtoH((void*) tab, dtab, sizeof(int) * prints);

    print_tab(tab + from, prints - from, prints,title, printer);
    cuCtxSynchronize();
    cuMemFreeHost(tab);
}

int ceil_div(int divident, int divisor) {
    if (!divident) return 0;
    return 1+ (divident-1) / divisor;
}

int ceil_div2(int divident, int power_of_2) {
    return 1+ ((divident-1) >> power_of_2);
}


CUdeviceptr cuAllocInts(int size) {
    CUdeviceptr result;
    cuMemAlloc(&result, sizeof(int)* size);
    return result;
}

int* cuAllocHostInts(int size) {
    int* result;
    cuMemAllocHost((void**) &result, sizeof(int)* size);
    return result;
}

CUdeviceptr addIntOffset(CUdeviceptr ptr, int offset) {
    return ptr + offset * sizeof(int);
}

int expand_to_power_of_2(int number, int power) {
    return ceil_div2(number, power) << power;
}

template <typename T>
T* cuMemAllocH(int size);

template<typename T>
T *cuMemAllocH(int size);

//template<typename T>
//T *cuMemAllocH(int size) {
//    T* result;
//    cuMemAllocHost((void**) &result, size * sizeof(T));
//    return result;
//}
