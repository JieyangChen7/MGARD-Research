/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_COMMON
#define MGRAD_CUDA_COMMON

#include <stdint.h>

namespace mgard_cuda {
enum error_bound_type { REL, ABS };
enum data_type { Float, Double };

using LENGTH = unsigned long long int;
using SIZE = unsigned int;
// using SIZE = int;
using DIM = uint32_t;
using QUANTIZED_INT = int;
using SERIALIZED_TYPE = unsigned char;

}

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Array.h"
#include "SubArray.h"
#include "Handle.h"
#include "Message.h"
#include "ErrorCalculator.h"
#include "MemoryManagement.h"

#endif
