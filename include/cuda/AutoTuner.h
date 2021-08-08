/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_AUTOTUNER
#define MGARD_CUDA_AUTOTUNER

#include "Task.h"
namespace mgard_cuda {

constexpr int GPK_CONFIG[5][7][3] = {{{1, 1, 8},{1, 1, 8},{1, 1, 8},{1, 1, 16},{1, 1, 32},{1, 1, 64},{1, 1, 128}},
                                     {{1, 2, 4},{1, 4, 4},{1, 4, 8},{1, 4, 16},{1, 4, 32},{1, 2, 64},{1, 2, 128}},
                                     {{2, 2, 2},{4, 4, 4},{4, 4, 8},{4, 4, 16},{4, 4, 32},{2, 2, 64},{2, 2, 128}},
                                     {{2, 2, 2},{4, 4, 4},{4, 4, 8},{4, 4, 16},{4, 4, 32},{2, 2, 64},{2, 2, 128}},
                                     {{2, 2, 2},{4, 4, 4},{4, 4, 8},{4, 4, 16},{4, 4, 32},{2, 2, 64},{2, 2, 128}}};


template <DIM D, typename T, typename DEVICE>
class AutoTuner {
public:
  MGARDm_CONT
  AutoTuner(Handle<D, T>& handle):handle(handle){};
  Handle<D, T>&handle;
};
}

#endif