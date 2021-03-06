/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <cstdint>
#include "cuda/Common.h"
#include "cuda/CompressionWorkflow.h"
#include "cuda/MemoryManagement.h"

#ifndef MGARD_API_CUDA_H
#define MGARD_API_CUDA_H

namespace mgard_cuda {

//!\file
//!\brief Compression and decompression API.

//! Compress a function on an N-D tensor product grid
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speed up compression.
//!\param[in] in_array Dataset to be compressed.
//!\param[in] type Error bound type: REL or ABS.
//!\param[in] tol Relative error tolerance.
//!\param[in] s Smoothness parameter to use in compressing the function.
//!
//!\return Compressed dataset.
template <uint32_t D, typename T>
Array<1, unsigned char> compress(Handle<D, T> &handle, Array<D, T> &in_array,
                                 enum error_bound_type type, T tol, T s);

//! Decompress a function on an N-D tensor product grid
//!
//!\param[in] handle Handle type for storing precomputed variable to
//! help speed up decompression.
//!\param[in] compressed_array Compressed dataset.
//!\return Decompressed dataset.
template <uint32_t D, typename T>
Array<D, T> decompress(Handle<D, T> &handle,
                       Array<1, unsigned char> &compressed_array);


void compress(std::vector<SIZE> shape, data_type T,
                double tol, double s, enum error_bound_type mode, 
                const void * original_data, 
                void *& compressed_data, size_t &compressed_size, Config config, bool isAllocated);

void decompress(const void * compressed_data, size_t compressed_size, 
                void *& decompressed_data, Config config, bool isAllocated);


bool verify(const void * compressed_data, size_t compressed_size);
enum data_type infer_type(const void * compressed_data, size_t compressed_size);
std::vector<SIZE> infer_shape(const void * compressed_data, size_t compressed_size);
} // namespace mgard_cuda

#endif