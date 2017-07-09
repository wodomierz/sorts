#include "bitonic/bitonic_sort.h"
#include "odd-even/odd_even.h"
#include "radix/radixsort.h"
#include "sample-rand/sample_rand.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

#include <ctime>
#include <cstring>
#include <algorithm>
#include <iostream>

#include <bitset>
#include <vector>

using namespace std;




vector<double> results1;
vector<double> results;


double res(vector<double>& results, vector<double>& results_opt) {
    vector<double> diffs;
    double r = 0;
    for (int i=1; i< results.size(); ++i) {
        diffs.push_back(results[i] - results_opt[i]);
    }
    sort(diffs.begin(), diffs.end());

    for (int i=1; i< diffs.size() -1; ++i) {
        r += diffs[i];
    }
    return r/(diffs.size()-2);
}


void print(int *tab, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d \n", tab[i]);
    }
}

typedef void (*func_t)(int *, int); // pointer to function with no args and void return
typedef double (*func_withtime)(int *, int);

double testTime(func_t f, int *c1, int n, string name) {
    std::clock_t start = std::clock();
    f(c1, n);
    double delta = (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000);

    std::cout << "Time for " << name << ": " << delta << " ms"
              << std::endl;
    return delta;
}

double cleanTestTime(func_t f, int const *init, int n, string name) {
    int *tab = (int *) malloc(n * sizeof(int));
    memcpy(tab, init, n * sizeof(int));
    double delta = testTime(f, tab, n, name);
    free(tab);
    return delta;
}

double cleanTest(func_withtime f, int const *init, int n, string name) {
    int *tab = (int *) malloc(n * sizeof(int));
    memcpy(tab, init, n * sizeof(int));
    double res = f(tab, n);
    free(tab);
    return res;
}



void cmpSorts(int n, int *initData) {
    double m1 = cleanTest([](int *tab, int size) -> double { return bitonic_sort(tab, size, true); }, initData, n, "odd-even");
    double m = cleanTest([](int *tab, int size) -> double { return bitonic_sort(tab, size, false); }, initData, n, "bitonic");
    results1.push_back(m1);
    results.push_back(m);
}

void cmpSorts1(int n, int *initData) {
    double m1 = cleanTestTime([](int *tab, int size) -> void { bitonic_sort(tab, size, true); }, initData, n, "bit");
    double m = cleanTestTime([](int *tab, int size) -> void {  std::sort(tab, tab + size); }, initData, n, "std");
    results1.push_back(m1);
    results.push_back(m);

}

void loggTitle(char const *title) {
    cout << "===================" << endl;
    cout << title << ":" << endl;
    cout << "===================" << endl;
    cout << endl;
}

void comparesorts(int n) {
    // int n = 1024*1024*50;
    cout << "Size " << n << ":" << endl;
    int *c1 = (int *) malloc(n * sizeof(int));


    //rand
    for (int j = 0; j < n; ++j) {
        c1[j] = rand();
    }
    loggTitle("rand");
    cmpSorts1(n, c1);
    // ascending
    for (int j = 0; j < n; ++j) {
        c1[j] = j;
    }
    loggTitle("asc");
    cmpSorts1(n, c1);

    //descending
    for (int j = 0; j < n; ++j) {
        c1[j] = n - j;
    }
    loggTitle("desc");
    cmpSorts1(n, c1);
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
//    for (int i = 1024*1024*256; i <= 1024*512*1024; i += 33456123) {
//        comparesorts(i);
//         comparesorts(i);
//    }
//    comparesorts(2048);

    for (int i = 0; i <= 2; i ++) {
        comparesorts(1024 * 1024 * 512);
    }



//    for (int i = 1024*1024; i <= 1024*64*1024; i += 33456123) {
//        comparesorts(i);
//        comparesorts(i);
//        comparesorts(i);
//    }
    cout << "FINAL bit radix over bitonic " << res(results, results1) << endl;
}


void testg(func_withtime sort, int n) {
    srand(time(NULL));
    int *c = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        c[j] = rand();
    }
//    for (int j = 0; j< n; ++j) {
//        c[j] = c[j]&((1<<31) - 1);
//    }
    sort(c, n);

    bool firsttime = true;

    for (int j = 0; j < (n - 1); ++j) {
        if (c[j] > c[j + 1]) {


//            std::bitset<32> first(c[j]);
//            std::cout << first << endl;
//            std::bitset<32> second(c[j+1]);
//            std::cout << second << endl;
            printf("test  %d %d %d %d \n",n, c[j], c[j + 1], j);
//            for (int i = 0 ; i < 32; ++i) {
//                cout << c[i] << endl;
////                std::bitset<32> b(c[i]);
////                std::cout << b << endl;
//            }
//            cout <<"dupeczka" << endl;
//            cout << endl << endl;
//            for (int i = 2048 ; i < 1064; ++i) {
//                cout << c[i] << endl;
////                std::bitset<32> b(c[i]);
////                std::cout << b << endl;
//            }
            printf("\n");
        }
//        if (!firsttime)
            assert(c[j] <= c[j + 1]);
        if (c[j] > c[j + 1]) {firsttime = false;}
    }
    printf("test %d ok\n",n);
    free(c);
}

void test_big(func_withtime sort);
void test01(func_withtime sort);

double radix1(int* tab, int size) {
    radixsort(tab, size);
}


double bitonic_sort1(int* tab, int size) {
    bitonic_sort(tab, size, true);
}
void test_correctness() {
//    test01(radix1);
//    testg(bitonic_sort, 1024);
//
    testg(bitonic_sort1 ,1024 * 2);
//    testg(bitonic_sort1 ,1024 * 4);
//    testg(bitonic_sort1 ,1024 * 8);
//    testg(bitonic_sort1 ,1024 * 16);
//    testg(bitonic_sort1 ,1024 * 32);
//    testg(bitonic_sort1 ,1024 * 1024);
//    testg(bitonic_sort1 ,1024 * 1024 *32);
//    testg(bitonic_sort1 ,512 * 1024 * 1024);
//    testg(bitonic_sort1 ,2 * 1024 * 1024 *1024);
//    testg(bitonic_sort1 ,1024 * 1024 * 512);
//    testg(bitonic_sort ,1024 * 16);
//    testg(bitonic_sort1 ,1024 * 23 * 512);
//    testg(radix1 ,1024 * 1024 * 512);
//
//    testg(radix1 ,10899);
//    testg(radix1 ,788068);
//    testg(radix1 ,607483);
//    test_big(radix1);
}

int main() {
//    ios_base::sync_with_stdio(false);


//
//    eff_tests();
    test_correctness();
    return 0;
}

void test01(func_withtime sort) {
    int n = 1024;
//    int n = 8;
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

void test0(func_withtime sort) {
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


void test_big(func_withtime sort) {
    int times = 1;
    int min = 1023;
    int max = 1024 * 1024;
    srand(time(NULL));
    while (times++ < 50) {
        int n = min + (rand() % (int) (max - min + 1));
        int *d = (int *) malloc(n * sizeof(int));
        for (int j = 0; j < n; ++j) {
            d[j] = rand();
        }
        for (int j = 0; j< n; ++j) {
            d[j] = d[j]&((1<<30) - 1);
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
