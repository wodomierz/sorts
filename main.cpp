#include "bitonic/bitonic_sort.h"
#include "odd-even/odd_even.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <ctime>
#include <cstring>
#include <algorithm>
#include <iostream>

using namespace std;





void print(int *tab, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d \n", tab[i]);
    }
}

typedef void (*func_t)(int *, int); // pointer to function with no args and void return

void testTime(func_t f, int *c1, int n, string name) {
    std::clock_t start = std::clock();
    f(c1, n);
    std::cout << "Time for " << name << ": " << (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000) << " ms"
              << std::endl;
}

void cleanTestTime(func_t f, int const *init, int n, string name) {
    int *tab = (int *) malloc(n * sizeof(int));
    memcpy(tab, init, n * sizeof(int));
    testTime(f, tab, n, name);
    free(tab);
}

void cmpSorts(int n, int *initData) {
    cleanTestTime([](int *tab, int size) -> void { bitonic_sort(tab, size); }, initData, n, "bitonic");
    cleanTestTime([](int *tab, int size) -> void { odd_even(tab, size); }, initData, n, "odd-even");
//    cleanTestTime([](int *tab, int size) -> void { sort(tab, tab + size); }, initData, n, "std");
    cleanTestTime([](int *tab, int size) -> void { odd_even1(tab, size); }, initData, n, "odd-even1");
}

void comparesorts(int n) {
    // int n = 1024*1024*50;
    cout << "Size " << n << ":" << endl;
    int *c1 = (int *) malloc(n * sizeof(int));

    //rand
    for (int j = 0; j < n; ++j) {
        c1[j] = rand();
    }

    cmpSorts(n, c1);
    // ascending
    for (int j = 0; j < n; ++j) {
        c1[j] = j;
    }
    cmpSorts(n, c1);
    //descending
    for (int j = 0; j < n; ++j) {
        c1[j] = n - j;
    }
    cmpSorts(n, c1);
}

void loggTitle(char const *title) {
    cout << "===================" << endl;
    cout << title << ":" << endl;
    cout << "===================" << endl;
    cout << endl;
}

void eff_tests() {
    srand(time(NULL));
//    loggTitle("multiples of 1024");
//    for (int i = 1; i <= 1024 * 64; i *= 2) {
//        comparesorts(i * 1024);
//    }
//    loggTitle("not multiples of 1024");
//    for (int i = 1; i < 1024 * 64; i *= 2) {
//        comparesorts(i * 1025);
//    }
//    loggTitle("big numbers");
//    for (int i = 1024 * 64; i <= 1024 * 256; i *= 2) {
//        comparesorts(i * 1024 - 1);
//    }
    for (int i = 1024*1024*256; i <= 1024*512*1024; i += 33456123) {
        comparesorts(i);
         comparesorts(i);
    }
}


void testg(func_t sort, int n) {
    int *c = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        c[j] = rand();
    }
    sort(c, n);

    for (int j = 0; j < (n - 1); ++j) {
        if (c[j] > c[j + 1]) {
            printf("test  %d %d %d %d \n",n, c[j], c[j + 1], j);
        }
        assert(c[j] <= c[j + 1]);
    }
    printf("test %d ok\n",n);
    free(c);
}

void test0(func_t sort);
void test_big(func_t sort);
//void test1();
//
//void test2();
//
//void test3();
void test01(func_t sort);

void test_correctness() {
    test0(odd_even1);

    testg(odd_even1 ,1024 * 2);
    testg(odd_even ,1024 * 23 * 512);
    testg(odd_even ,1024 * 1024 * 512);

    testg(odd_even ,10899);
    testg(odd_even ,788068);
    testg(odd_even ,607483);
    test_big(odd_even);
}

int main() {
//    ios_base::sync_with_stdio(false);


//
    eff_tests();
//    test_correctness();
    return 0;
}

void test01(func_t sort) {
//    int n = 1024;
    int n = 16;
    int *c = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        c[j] = n - j;
    }
    sort(c, n);
    for (int j = 0; j < (n - 1); ++j) {
        if (c[j] > c[j + 1]) {
            printf("test01");
            for (int i = 0 ; i < n; ++i) {
                printf(" %d", c[i]);
            }
            printf("\n");

        }
        assert(c[j] <= c[j + 1]);
        if (c[j] + 1 != c[j + 1]) {
            printf("test0 %d %d\n", c[j], c[j + 1]);

        }
        assert(c[j] + 1 == c[j + 1]);
    }
    printf("test0 ok\n");
    free(c);
}

void test0(func_t sort) {
//    int n = 1024;
    int n = 16;
    int *c = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        c[j] = n - j;
    }
    sort(c, n);
    for (int j = 0; j < (n - 1); ++j) {
        if (c[j] > c[j + 1]) {
            printf("test0 %d %d %d\n",j, c[j], c[j + 1]);

        }
        assert(c[j] <= c[j + 1]);
        if (c[j] + 1 != c[j + 1]) {
            printf("test0 %d %d\n", c[j], c[j + 1]);

        }
        assert(c[j] + 1 == c[j + 1]);
    }
    printf("test0 ok\n");
    free(c);
}

void test1(func_t sort) {
    int n = 1024 * 1024 * 512;
    int *c = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        c[j] = rand();
    }
    sort(c, n);
    for (int j = 0; j < (n - 1); ++j) {
        if (c[j] > c[j + 1]) {
            printf("test1 %d %d\n", c[j], c[j + 1]);
        }
        assert(c[j] <= c[j + 1]);
    }
    printf("test1 ok\n");
    free(c);
}


void test2(func_t sort) {
    int n = 1024 * 23;
    int *c = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        c[j] = rand();
    }
    bitonic_sort(c, n);

    for (int j = 0; j < (n - 1); ++j) {
        if (c[j] > c[j + 1]) {
            printf("test2 %d %d %d \n", c[j], c[j + 1], j);
        }
         assert(c[j] <= c[j + 1]);
    }
    printf("test2 ok\n");
    free(c);
}

void test3() {
    int n = 10899;
    int *d = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        d[j] = rand();
    }
    bitonic_sort(d, n);
    for (int j = 0; j < (n - 1); ++j) {
        if (d[j] > d[j + 1]) {
            printf("test3 %d %d\n", d[j], d[j + 1]);
        }
        assert(d[j] <= d[j + 1]);
    }
    printf("test3 ok\n");
    free(d);
}

void test_big(func_t sort) {
    int times = 1;
    int min = 1023;
    int max = 1024 * 1024;

    while (times++ < 50) {
        int n = min + (rand() % (int) (max - min + 1));
        int *d = (int *) malloc(n * sizeof(int));
        for (int j = 0; j < n; ++j) {
            d[j] = rand();
        }
        sort(d, n);
        for (int j = 0; j < (n - 1); ++j) {
            if (d[j] > d[j + 1]) {
                printf("testbig %d times %d %d %d %d\n", n, times, j, d[j], d[j + 1]);
            }
            assert(d[j] <= d[j + 1]);
        }
        free(d);
    }

    printf("testbig ok\n");
}
