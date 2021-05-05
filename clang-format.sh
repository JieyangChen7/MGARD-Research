#!/bin/sh

clang-format -i ./src/cuda/*.cu
clang-format -i ./src/cuda/*.cpp
clang-format -i ./src/cuda/GridProcessingKernels/*.cu
clang-format -i ./src/cuda/LinearProcessingKernels/*.cu
clang-format -i ./src/cuda/IterativeProcessingKernels/*.cu
clang-format -i ./src/cuda/LevelwiseProcessingKernels/*.cu
clang-format -i ./src/cuda/LinearQuantization/*.cu
clang-format -i ./src/cuda/ParallelHuffman/*.cu
clang-format -i ./src/cuda/ParallelHuffman/*.cc

clang-format -i ./include/cuda/*.h
clang-format -i ./include/cuda/*.hpp
clang-format -i ./include/cuda/ParallelHuffman/*.hh
clang-format -i ./include/cuda/ParallelHuffman/*.cuh
clang-format -i ./include/mgard_cuda_api.h
clang-format -i ./include/mgard_api.h

clang-format -i ./tests/gpu-cuda/*.cpp

./scripts/travis/run-clang-format.sh
