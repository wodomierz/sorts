CUDA_INSTALL_PATH ?= /usr/local/cuda
VER := -5

CXX := /usr/bin/g++$(VER)
CC := /usr/bin/gcc$(VER)
LINK := /usr/bin/g++$(VER) -fPIC
CCPATH := ./gcc$(VER)
NVCC  := $(CUDA_INSTALL_PATH)/bin/nvcc -ccbin $(CCPATH)

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Libraries
LIB_CUDA := -lcuda


# Options
NVCCOPTIONS = -arch sm_30 -ptx

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS) $(NVCCOPTIONS)
CXXFLAGS += $(COMMONFLAGS) -std=c++11
CFLAGS += $(COMMONFLAGS)



CUDA_OBJS = bitonic/bitonic_sort.ptx odd-even/odd_even.ptx radix/radixsort.ptx \
sample-rand/sample_rand.ptx
OBJS =  main.cpp.o bitonic/bitonic_sort.cpp.o odd-even/odd_even.cpp.o utils/utils.cpp.o \
radix/radixsort.cpp.o sample-rand/sample_rand.cpp.o \
sample-rand/sample_rand_context.cpp.o sample-rand/sample_rand_device.cpp.o sample-rand/PrefsumContext.cpp.o

TARGET = solution.x
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES:	.c	.cpp	.cu	.o	
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): prepare $(OBJS) $(CUDA_OBJS)
	$(LINKLINE)


clean:
	rm -rf -r $(TARGET) *.o *.ptx
	find . -type f -name '*.o' -delete
	find . -type f -name '*.ptx' -delete

prepare:
	rm -rf $(CCPATH);\
	mkdir -p $(CCPATH);\
	ln -s $(CXX) $(CCPATH)/g++;\
	ln -s $(CC) $(CCPATH)/gcc;

