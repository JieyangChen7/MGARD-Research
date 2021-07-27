/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: Jul 20, 2021
 */

#ifndef MGARD_CUDA_SUBARRAY
#define MGARD_CUDA_SUBARRAY
#include "Common.h"
#include <vector>

namespace mgard_cuda {

template <DIM D, typename T> class SubArray {
public:
  SubArray();
  SubArray(Array<D, T> &array);
  SubArray(std::vector<SIZE> shape, T * dv, std::vector<SIZE> ldvs_h, SIZE * ldvs_d);
  SubArray(SubArray<D, T> &subArray);
  void offset(std::vector<SIZE> idx);
  void resize(std::vector<SIZE> shape);
  void offset(DIM dim, SIZE offset_value);
  void resize(DIM dim, SIZE new_size);
  void project(DIM dim0, DIM dim1, DIM dim2);
  ~SubArray();

  T *dv;
  std::vector<SIZE> ldvs_h;
  SIZE *ldvs_d;
  std::vector<SIZE> shape;
  DIM projected_dim0;
  DIM projected_dim1;
  DIM projected_dim2;
  SIZE lddv1;
  SIZE lddv2;
};

} // namespace mgard_cuda
#endif