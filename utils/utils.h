//
// Created by Miron Ficak on 28/06/2017.
//


#ifndef SORTS_UTILS_H
#define SORTS_UTILS_H

#include <string>
#include "cuda.h"
#include <vector>

#define PRINT(...) {}
#define PRINT1(...) fprintf(stderr, ##__VA_ARGS__)

#define MAX_GRID_DIM (1 << 30)

void manageError(CUresult, int, const char *);
void manageErrorWithMessage(CUresult, std::string, int, const char *);

//typedef std::vector<std::string> String;
typedef bool (*Filter)(int, int *);

static Filter everything = [](int i, int *val) -> bool { return true; };

typedef void (*Printer)(int, int *);

static Printer simplePrint = [](int i, int *tab) { PRINT1("%d ", tab[i]); };
static Printer indexedPrint = [](int i, int *tab) { PRINT1("(%d %d) ", i, tab[i]); };
#define assertPrintable(print, condition) ({if (!(condition)) {print(); assert(false);}})

void
print_tab(int *tab, int size, int prints, const char *title, Printer printer = simplePrint, Filter filter = everything);

void print_Devtab(CUdeviceptr &dtab, int size, int prints, int from = 0, const char *title = "",
                  Printer printer = simplePrint, Filter filter = everything);

#define manageResultWithMessage(error_code, comment) (manageErrorWithMessage((error_code), (comment), __LINE__, __FILE__))
#define manageResult(error_code) (manageError((error_code), __LINE__, __FILE__))
#define safeLaunch1Dim(f, x, y, threads, args) (manageResult(cuLaunchKernel((f), (x),(y),1,(threads),1,1,0,0,(args),0)))

int ceil_div(int, int);

CUdeviceptr cuAllocInts(int size);

template<typename T>
T *cuMemAllocH(int size) {
    T *result;
    cuMemAllocHost((void **) &result, size * sizeof(T));
    return result;
}


int *cuAllocHostInts(int size);

template<typename T>
CUdeviceptr cuAllocD(int size) {
    CUdeviceptr ptr;
    cuMemAlloc(&ptr, size * sizeof(T));
    return ptr;
}


CUdeviceptr addIntOffset(CUdeviceptr ptr, int offset);

int expand_to_power_of_2(int number, int power);

#endif //SORTS_UTILS_H

