/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <vector>

 
#include "cuda/CommonInternal.h"

#include "cuda/CompressionWorkflow.h"

#include "cuda/MemoryManagement.h"

#include "cuda/DataRefactoring.h"
#include "cuda/LinearQuantization.h"
#include "cuda/LosslessCompression.h"


namespace mgard_cuda {

template <DIM D, typename T>
void compress(std::vector<SIZE> shape,
                T tol, T s, enum error_bound_type mode, 
                const void * original_data, 
                void * compressed_data, size_t &compressed_size, Config config) {
  Handle<D, T> handle(shape, config);
  mgard_cuda::Array<D, T> in_array(shape);
  in_array.loadData((const T*)original_data);
  Array<1, unsigned char> compressed_array =
      compress(handle, in_array, mode, tol, s);
  compressed_size = compressed_array.getShape()[0];
  std::memcpy(compressed_data, compressed_array.getDataHost(),
         compressed_size);  
}


template <DIM D, typename T>
void decompress(std::vector<SIZE> shape,
                const void * compressed_data, size_t compressed_size,
                void * decompressed_data) {
  size_t original_size = 1;
  for (int i = 0; i < D; i++) original_size *= shape[i];
  Handle<D, T> handle(shape);
  std::vector<SIZE> compressed_shape(1); 
  compressed_shape[0] = compressed_size;
  Array<1, unsigned char> compressed_array (compressed_shape);
  compressed_array.loadData((const unsigned char*)compressed_data);
  Array<D, T> out_array =
      decompress(handle, compressed_array);
  std::memcpy(decompressed_data, out_array.getDataHost(),
         original_size * sizeof(T));
}

void compress(std::vector<SIZE> shape, data_type T,
                double tol, double s, enum error_bound_type mode, 
                const void * original_data, 
                void * compressed_data, size_t &compressed_size, Config config) {
  if (shape.size() == 1) {
    if (T == Double) {
        mgard_cuda::compress<1, double>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    } else {
        mgard_cuda::compress<1, float>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    }
  } else if (shape.size() == 2){
    if (T == Double) {
        mgard_cuda::compress<2, double>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    } else {
        mgard_cuda::compress<2, float>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    }
  } else if (shape.size() == 3){
    if (T == Double) {
        mgard_cuda::compress<3, double>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    } else {
        mgard_cuda::compress<3, float>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    }
  } else if (shape.size() == 4){
    if (T == Double) {
        mgard_cuda::compress<4, double>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    } else {
        mgard_cuda::compress<4, float>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    }
  } else if (shape.size() == 5){
    if (T == Double) {
        mgard_cuda::compress<5, double>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    } else {
        mgard_cuda::compress<5, float>(shape, tol, s, mode,
                                   original_data, compressed_data, compressed_size, config);
    }
  }
}


void decompress(std::vector<SIZE> shape, data_type T,
                  const void * compressed_data, size_t compressed_size, 
                  void * decompressed_data) {
  if (shape.size() == 1) {
    if (T == Double) {
        mgard_cuda::decompress<1, double>(shape,
                                   compressed_data, compressed_size, decompressed_data);
    } else {
        mgard_cuda::decompress<1, float>(shape,
                                   compressed_data, compressed_size, decompressed_data);
    }
  } else if (shape.size() == 2){
    if (T == Double) {
        mgard_cuda::decompress<2, double>(shape,
                                   compressed_data, compressed_size, decompressed_data);
    } else {
        mgard_cuda::decompress<2, float>(shape,
                                   compressed_data, compressed_size, decompressed_data);
    }
  } else if (shape.size() == 3){
    if (T == Double) {
        mgard_cuda::decompress<3, double>(shape,
                                   compressed_data, compressed_size, decompressed_data);
    } else {
        mgard_cuda::decompress<3, float>(shape, 
                                   compressed_data, compressed_size, decompressed_data);
    }
  } else if (shape.size() == 4){
    if (T == Double) {
        mgard_cuda::decompress<4, double>(shape, 
                                   compressed_data, compressed_size, decompressed_data);
    } else {
        mgard_cuda::decompress<4, float>(shape, 
                                   compressed_data, compressed_size, decompressed_data);
    }
  } else if (shape.size() == 5){
    if (T == Double) {
        mgard_cuda::decompress<5, double>(shape, 
                                   compressed_data, compressed_size, decompressed_data);
    } else {
        mgard_cuda::decompress<5, float>(shape, 
                                   compressed_data, compressed_size, decompressed_data);
    }
  }

}


bool verify(data_type T, const void * compressed_data, size_t compressed_size) {
  char signature[SIGNATURE_SIZE];
  if (T == Double) {
    if (compressed_size < sizeof(quant_meta<double>)) return false;
    const quant_meta<double> * meta = (const quant_meta<double>*)compressed_data; 
    strncpy(signature, meta->signature, SIGNATURE_SIZE);
  } else {
    if (compressed_size < sizeof(quant_meta<float>)) return false;
    const quant_meta<float> * meta = (const quant_meta<float>*)compressed_data; 
    strncpy(signature, meta->signature, SIGNATURE_SIZE);
  }
  if (strcmp(signature, SIGNATURE) == 0) {
    return true;
  } else {
    return false;
  }
}

#define KERNELS(D, T)                                 \
  template void compress<D, T>(std::vector<SIZE> shape, \
                  T tol, T s, enum error_bound_type mode, \
                  const void * original_data, \
                  void * compressed_data, size_t &compressed_size, Config config); \
  template void decompress<D, T>(std::vector<SIZE> shape, \
                  const void * compressed_data, size_t compressed_size, \
                  void * decompressed_data);

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)
KERNELS(4, double)
KERNELS(4, float)
KERNELS(5, double)
KERNELS(5, float)
#undef KERNELS



}
