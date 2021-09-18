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

class Device {};
class CUDA: public Device {};
class HIP: public Device {};
class DPCxx: public Device {};
class OpenMp: public Device {};
class Kokkos: public Device {};

enum error_bound_type { REL, ABS };
// enum data_type { Float, Double };
enum class data_type:uint8_t { Float, Double };

using IDX = unsigned long long int;
using LENGTH = unsigned long long int;
using SIZE = unsigned int;
// using SIZE = int;
using DIM = uint32_t;
using QUANTIZED_INT = int;
using SERIALIZED_TYPE = unsigned char;
using Byte = unsigned char;
using OPTION = int8_t;
}

#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Array.h"
#include "Handle.h"
#include "Message.h"
#include "ErrorCalculator.h"
#include "MemoryManagement.h"

#endif
