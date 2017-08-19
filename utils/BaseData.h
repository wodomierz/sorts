//
// Created by Miron Ficak on 03/08/2017.
//

#ifndef SORTS_BASE_DATA_H
#define SORTS_BASE_DATA_H


#include "utils.h"

struct BaseData {
public:
    int x_dim;
    int y_dim;
    int number_of_blocks;
    int size;

    BaseData(int size, int block_size) : size(size) {
        number_of_blocks = ceil_div(size, block_size);
        x_dim = number_of_blocks > MAX_GRID_DIM ? MAX_GRID_DIM : number_of_blocks;
        y_dim = ceil_div(number_of_blocks, x_dim);
    }
};


#endif //SORTS_BASE_DATA_H
