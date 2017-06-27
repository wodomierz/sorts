//
// Created by Miron Ficak on 28/06/2017.
//
#include <string>
#include "cuda.h"
#ifndef SORTS_UTILS_H
#define SORTS_UTILS_H
void manageError(CUresult, std::string, int, const char*);
#define manageResult(error_code, comment) (manageError((error_code), (comment), __LINE__, __FILE__))
#endif //SORTS_UTILS_H

