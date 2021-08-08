/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGARD_CUDA_FUNCTOR
#define MGARD_CUDA_FUNCTOR

namespace mgard_cuda {

template <DIM D, typename T>
class Functor {
public:
  // MGARDm_CONT Functor() {}
  // MGARDm_EXEC void operator()(IDX ngridz, IDX ngridy, IDX ngirdx,
  //              IDX nblockz, IDX nblocky, IDX nblockx,
  //              IDX blockz, IDX blocky, IDX blockx,
  //              IDX threadz, IDX thready, IDX threadx, T * shared_memory){
  // }

  MGARDm_EXEC void
  __operation1(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) {
  }

  MGARDm_EXEC void
  __operation2(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) {
  }

  MGARDm_EXEC void
  __operation3(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) {
  }

};
}

#endif