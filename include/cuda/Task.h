/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include "Functor.h"

#ifndef MGARD_CUDA_TASK
#define MGARD_CUDA_TASK

namespace mgard_cuda {

template <DIM D, typename T, typename FUNCTOR>
class Task {
public:
    MGARDm_CONT
    Task(FUNCTOR functor, 
         IDX ngridz, IDX ngridy, IDX ngridx,
         IDX nblockz, IDX nblocky, IDX nblockx,
         LENGTH shared_memory_size,
         int queue_idx): 
          functor(functor), 
          ngridz(ngridz), ngridy(ngridy), ngridx(ngridx),
          nblockz(nblockz), nblocky(nblocky), nblockx(nblockx),
          shared_memory_size(shared_memory_size),
          queue_idx(queue_idx) {}

    MGARDm_EXEC void 
    __operation1(IDX ngridz, IDX ngridy, IDX ngridx, 
                    IDX nblockz, IDX nblocky, IDX nblockx,
                    IDX blockz, IDX blocky, IDX blockx, 
                    IDX threadz, IDX thready, IDX threadx,
                    T * shared_memory) {
      functor.__operation1(ngridz, ngridy, ngridx,
             nblockz, nblocky, nblockx,
             blockz, blocky, blockx,
             threadz, thready, threadx, shared_memory);
    }

    MGARDm_EXEC void 
    __operation2(IDX ngridz, IDX ngridy, IDX ngridx, 
                    IDX nblockz, IDX nblocky, IDX nblockx,
                    IDX blockz, IDX blocky, IDX blockx, 
                    IDX threadz, IDX thready, IDX threadx,
                    T * shared_memory) {
      functor.__operation2(ngridz, ngridy, ngridx,
             nblockz, nblocky, nblockx,
             blockz, blocky, blockx,
             threadz, thready, threadx, shared_memory);
    }

    MGARDm_EXEC void 
    __operation3(IDX ngridz, IDX ngridy, IDX ngridx, 
                    IDX nblockz, IDX nblocky, IDX nblockx,
                    IDX blockz, IDX blocky, IDX blockx, 
                    IDX threadz, IDX thready, IDX threadx,
                    T * shared_memory) {
      functor.__operation3(ngridz, ngridy, ngridx,
             nblockz, nblocky, nblockx,
             blockz, blocky, blockx,
             threadz, thready, threadx, shared_memory);
    }
    MGARDm_CONT int get_queue_idx() {return queue_idx;}
    MGARDm_CONT IDX get_ngridz() {return ngridz;}
    MGARDm_CONT IDX get_ngridy() {return ngridy;}
    MGARDm_CONT IDX get_ngridx() {return ngridx;}
    MGARDm_CONT IDX get_nblockz() {return nblockz;}
    MGARDm_CONT IDX get_nblocky() {return nblocky;}
    MGARDm_CONT IDX get_nblockx() {return nblockx;}
    MGARDm_CONT LENGTH get_shared_memory_size () {return shared_memory_size; }
  private:
    FUNCTOR functor;
    IDX ngridz, ngridy, ngridx;
    IDX nblockz, nblocky, nblockx;
    LENGTH shared_memory_size;
    int queue_idx;
};

}
#endif