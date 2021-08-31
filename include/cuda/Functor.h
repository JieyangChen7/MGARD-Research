/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_FUNCTOR
#define MGARD_CUDA_FUNCTOR

namespace mgard_cuda {

template <typename DeviceType>
class Functor {
public:
  MGARDm_EXEC void
  Init(IDX ngridz, IDX ngridy, IDX ngridx,
       IDX nblockz, IDX nblocky, IDX nblockx,
       IDX blockz, IDX blocky, IDX blockx,
       IDX threadz, IDX thready, IDX threadx, Byte * shared_memory) {
    this->ngridz = ngridz; this->ngridy = ngridy; this->ngridx = ngridx;
    this->nblockz = nblockz; this->nblocky = nblocky; this->nblockx = nblockx;
    this->blockz = blockz; this->blocky = blocky; this->blockx = blockx;
    this->threadz = threadz; this->thready = thready; this->threadx = threadx;
    this->shared_memory = shared_memory;
  }

  MGARDm_EXEC void
  Operation1();

  MGARDm_EXEC void
  Operation2();

  MGARDm_EXEC void
  Operation3();

  MGARDm_EXEC void
  Operation4();

  MGARDm_EXEC void
  Operation5();

  IDX ngridz, ngridy, ngridx;
  IDX nblockz, nblocky, nblockx;
  IDX blockz, blocky, blockx;
  IDX threadz, thready, threadx;
  Byte * shared_memory;
};
}

#endif