.PHONY: all run

SRCS=main.cu
NVCCFLAGS=-Wno-deprecated-gpu-targets -use_fast_math -O3

all: $(SRCS)
	nvcc $(NVCCFLAGS) $(SRCS)
