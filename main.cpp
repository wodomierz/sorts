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
#include <cmath>
#include "quick-sort/quick_sort.h"

using namespace std;


struct result {
    int amount;
    double time;
};

vector<double> results1;
vector<double> results;

vector<vector<double>> allres;
vector<long> avgs;

//bit, odd-oddeven, radix, quick

double res(vector<double> &results, vector<double> &results_opt) {
    //res
    vector<double> diffs;
    double r = 0;
    for (int i = 1; i < results.size(); ++i) {
        diffs.push_back(results[i] - results_opt[i]);
    }
    sort(diffs.begin(), diffs.end());

    for (int i = 1; i < diffs.size() - 1; ++i) {
        r += diffs[i];
    }
    return r / (diffs.size() - 2);
}


void print(int *tab, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d eq \n", tab[i]);
    }
}

typedef void (*func_t)(int *, int); // pointer to function with no args and void return
void standard_sort(int*tab, int size) {
    std::sort(tab, tab + size);
}

vector<func_t > sorts = {bitonic_sort, odd_even, radixsort, quick_sort};

typedef void (data_provider)(int*, int);

typedef double (*func_withtime)(int *, int);

double testTime(func_t f, int *c1, int n, string name) {
    std::clock_t start = std::clock();
    f(c1, n);
    double delta = (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000);

//    std::cout << "Time for " << name << ": " << delta << " ms"
//              << std::endl;
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


//void cmpSorts(int n, int *initData) {
//    double m1 = cleanTest([](int *tab, int size) -> double { return bitonic_sort(tab, size); }, initData, n,
//                          "odd-even");
//    double m = cleanTest([](int *tab, int size) -> double { return bitonic_sort(tab, size); }, initData, n, "bitonic");
//    results1.push_back(m1);
//    results.push_back(m);
//}

void testSort(int *initData, int n, int sort_idx) {
    func_t sort = sorts[sort_idx];
    cleanTestTime(sort, initData, n, "sort");
    allres[sort_idx].push_back(cleanTestTime(sort, initData, n, "sort"));
}

void testStandardSort(int *initData, int n) {
    allres[4].push_back(cleanTestTime(standard_sort, initData, n, "sort"));
}

long avg(vector<double> v) {
    double acc = 0;
    for (double value: v) {
        acc += value;
    }
    return std::lround(acc / v.size());
}

void testAvg(int *initData, int n, data_provider provider) {
    int times = 8;
    int parallel_times = 3;

    allres.clear();
    for (int i=0; i< sorts.size() + 1; ++i) {
        allres.push_back(vector<double>());
    }
    avgs.resize(sorts.size());
    while(times--) {
        provider(initData, n);
        for (int round=0; round< parallel_times; ++round) {
            for (int i = 0; i < sorts.size(); ++i) {
                testSort(initData, n, i);
            }
        }
        testStandardSort(initData, n);
    }
    for(int i =0;i < sorts.size(); ++i) {
        avgs[i] = avg(allres[i]);
    }
    avgs[4] = avg(allres[4]);

    cout << "BIT " << avgs[0] << endl;
    cout << "ODD_EVEN "<< avgs[1]<< endl;
    cout<<"RADIX "<< avgs[2]<<endl;
    cout<<"QUICK "<< avgs[3]<< endl;
    cout<<"STD " << avgs[4]<< endl;
}


void randDP(int* tab, int size) {
    srand(time(NULL));
    for (int j = 0; j < size; ++j) {
        tab[j] = rand();
    }
}
void ascDP(int* tab, int size) {
    for (int j = 0; j < size; ++j) {
        tab[j] = j;
    }
}
void descDP(int* tab, int size) {
    for (int j = 0; j < size; ++j) {
        tab[j] = size - j;
    }
}





void cmpSorts1(int n, int *initData) {
    double m1 = cleanTestTime([](int *tab, int size) -> void { quick_sort(tab, size); }, initData, n, "quick");
    m1 = cleanTestTime([](int *tab, int size) -> void { quick_sort(tab, size); }, initData, n, "quick");
    double m = cleanTestTime([](int *tab, int size) -> void { std::sort(tab, tab + size); }, initData, n, "std");
    results1.push_back(m1);
    results.push_back(m);
}

void loggTitle(char const *title) {
    cout << "===================" << endl;
    cout << title << ":" << endl;
    cout << "===================" << endl;
    cout << endl;
}

void allTests(int n) {
    cout <<endl;
    cout << "Size " << n << ":" << endl;
    int *tab = (int *) malloc(n * sizeof(int));

    loggTitle("rand");
    testAvg(tab, n, randDP);
    loggTitle("asc");
    testAvg(tab, n, ascDP);

    loggTitle("desc");
    testAvg(tab, n, descDP);
    free(tab);
}

void effTests() {
    for (int i= 20; i <= 29; ++i) {
        allTests((1 << i));
    }
    cout << "END" << endl;
}

void comparesorts(int n, int times = 1) {
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
//    losggTitle("multiples of 1024");
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

    for (int i = 0; i <= 2; i++) {
        comparesorts(1 << 28);
    }



//    for (int i = 1024*1024; i <= 1024*64*1024; i += 33456123) {
//        comparesorts(i);
//        comparesorts(i);
//        comparesorts(i);
//    }
    cout << "FINAL bit radix over bitonic " << res(results, results1) << endl;
}


void testg(func_t sort, int n) {
    srand(time(NULL));
    int *c = (int *) malloc(n * sizeof(int));
    for (int j = 0; j < n; ++j) {
        c[j] = n - j - 1;
//                rand();
    }
//    for (int j = 0; j< n; ++j) {
//        c[j] = c[j]&((1<<31) - 1);
//    }
    sort(c, n);
    int bucket_size = n / 1024;
//    for (int r = 0; r < n; r += bucket_size) {
//        std::sort(c +  r , c + r + bucket_size);
//    }

    bool firsttime = true;

    for (int j = 0; j < (n - 1); ++j) {
        if (c[j] != j) {


//            std::bitset<32> first(c[j]);
//            std::cout << first << endl;
//            std::bitset<32> second(c[j+1]);
//            std::cout << second << endl;
            printf("test %d %d %d %d \n", n, c[j], c[j + 1], j);
            for (int i = -2; (i + j) < n && i < 10; ++i) {
                cout << (i + j) << " " << c[i + j] << endl;
////                std::bitset<32> b(c[i]);
////                std::cout << b << endl;
            }
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
        assert(c[j] == j);
        if (c[j] > c[j + 1]) { firsttime = false; }
    }
    printf("test %d ok\n", n);
    free(c);
}

void test_big(func_withtime sort);

void test01(func_withtime sort);

double radix1(int *tab, int size) {
    radixsort(tab, size);
}


void bitonic_sort1(int *tab, int size) {
 odd_even(tab, size);
//    sampleRand(tab, size);
//    quick_sort(tab, size);
//    odd_even(tab,size);
}

void test_correctness() {
    srand(time(NULL));
//    testg(bitonic_sort1, 1024*4);
//    testg(bitonic_sort1, 1024*1024);
//    testg(bitonic_sort1, 1024*512);
//    testg(bitonic_sort1, 1024*1024*2);
//    testg(bitonic_sort1, 1024*1024*2);
//    testg(bitonic_sort1, 1024*1024*2);
//    testg(bitonic_sort1, 1024*2);
//    testg(bitonic_sort1, 1024*4);
//    testg(bitonic_sort1, 1024*8);
//    testg(bitonic_sort1, 1024*1024*2);
//    testg(bitonic_sort1, 1024*1024*4);
//    testg(bitonic_sort1,16*16*2);
//    testg(bitonic_sort1, 16*16*4);
//    testg(bitonic_sort1, 16*16*8);
//    testg(bitonic_sort1, 1024*1024*8);
//    testg(bitonic_sort1, 1024*1024*16);
//    testg(bitonic_sort1, 1024*1024*32);
//    testg(bitonic_sort1, 1024*1024*216);
//    testg(bitonic_sort1, 1024*1024*32);
    int z = 1000;
    while(z--) {
        testg(bitonic_sort1, 1<<28);
    }
//    testg(bitonic_sort1, 1024*1024*32);
//    testg(bitonic_sort1, 1024*1024*64);
//    test_big(bitonic_sort1);
//    testg(bitonic_sort1, 1024*1024*128);
//    testg(bitonic_sort1, 1024*1024*128);
//    testg(bitonic_sort1, 1024*1024*128);
//    testg(bitonic_sort1, S_SIZE * BLOCK_SIZE);
//    testg(bitonic_sort1, S_SIZE * S_SIZE * S_SIZE * S_SIZE * BLOCK_SIZE);
//    testg(bitonic_sort1 , 1024*1024);
//    int z =30;
//    while(--z) {
//        int max = 600*1024*1024;
//        int min = 300000000;
//        int i =  rand()%(max-min + 1) + min;
//
//        testg(bitonic_sort1 , i);
//    }
//    testg(bitonic_sort1, 123);
//    testg(bitonic_sort1, 1024*234);
//    testg(bitonic_sort1 , 1024*1024);
//    testg(bitonic_sort1 , 234567890);
}

int main() {
//    ios_base::sync_with_stdio(false);


//
//    effTests();
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
            for (int i = 0; i < n; ++i) {
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
            printf("test0 %d %d %d\n", j, c[j], c[j + 1]);

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
        for (int j = 0; j < n; ++j) {
            d[j] = d[j] & ((1 << 30) - 1);
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
