//
// Created by Miron Ficak on 28/06/2017.
//
#include <string>
#include <functional>
#include "cuda.h"

#define PRINT(...) {}
#define PRINT1(...) fprintf(stderr, ##__VA_ARGS__)

#ifndef SORTS_UTILS_H
#define SORTS_UTILS_H

#define MAX_GRID_DIM (1 << 30)

void manageError(CUresult, std::string, int, const char*);

typedef std::function<bool(int, int*)> Filter;
static Filter everything = [](int i, int* val) -> bool {return true;};
typedef std::function<void(int, int*)> Printer;
static Printer simplePrint = [](int i, int* tab) {PRINT1("%seq ", tab[i]);};
static Printer indexedPrint = [](int i, int* tab) {PRINT1("(%seq %seq) ", i, tab[i]);};
#define assertPrintable(print, condition) ({if (!(condition)) {print(); assert(false);}})
void print_tab(int* tab, int size, int prints, const char* title, Printer printer = simplePrint, Filter filter = everything);
void print_Devtab(CUdeviceptr& dtab, int size, int prints, int from = 0, const char* title = "", Printer printer = simplePrint, Filter filter = everything);
#define manageResult(error_code, comment) (manageError((error_code), (comment), __LINE__, __FILE__))

int ceil_div(int, int);
CUdeviceptr cuAllocInts(int size);
template <typename T>
T* cuMemAllocH(int size);
int* cuAllocHostInts(int size);
template <typename T>
CUdeviceptr cuAllocD(int size) {
    CUdeviceptr ptr;
    cuMemAlloc(&ptr, size * sizeof(T));
    return ptr;
}


CUdeviceptr addIntOffset(CUdeviceptr ptr, int offset);

int expand_to_power_of_2(int number, int power);
#endif //SORTS_UTILS_H

