//
// Created by Miron Ficak on 28/06/2017.
//

#include "utils.h"

void manageError(CUresult res, std::string message, int line, const char* file) {
    if (res != CUDA_SUCCESS) {
        printf("%s , error nr: %d file : %s line: %d\n", message.c_str(), res, file, line);
        exit(1);
    }
}