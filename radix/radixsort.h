#if !defined(RADIX_SORT_H)
#define RADIX_SORT_H 1


#define RADIX_THREADS_POW 10
#define RADIX_THREADS (1 << RADIX_THREADS_POW)
#define RADIX_ELEMENTS 4
#define RADIX_BLOCK (RADIX_THREADS * RADIX_ELEMENTS)

void radixsort(int *, int);


#endif


