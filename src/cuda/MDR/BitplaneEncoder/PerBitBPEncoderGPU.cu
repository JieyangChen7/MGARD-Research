/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */
#include "cuda/CommonInternal.h"
 
#include "cuda/MDR/BitplaneEncoder/PerBitBPEncoderGPU.hpp"

namespace mgard_cuda {
namespace MDR {

#define KERNELS(D, T) template class PER_BIT_ENCODER_AutoTuner<D, T, CUDA>;

KERNELS(1, double)
KERNELS(1, float)
KERNELS(2, double)
KERNELS(2, float)
KERNELS(3, double)
KERNELS(3, float)

#undef KERNELS

}
} // namespace mgard_cuda