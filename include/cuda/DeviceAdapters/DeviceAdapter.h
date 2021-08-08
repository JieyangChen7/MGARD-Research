/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */
#ifndef MGARD_CUDA_DEVICE_ADAPTER
#define MGARD_CUDA_DEVICE_ADAPTER

namespace mgard_cuda {


template <typename DEVICE> 
MGARDm_EXEC void SyncThreads() {
  if (std::is_same<DEVICE, CUDA>::value) {
    __syncthreads();
  }
}

template <DIM D, typename T, typename TASK, typename DEVICE>
class DeviceAdapter {
public:
  MGARDm_CONT
  DeviceAdapter(Handle<D, T>& handle):handle(handle){};
  MGARDm_CONT
  void Execute() {};
  Handle<D, T>& handle;
};


}

#endif