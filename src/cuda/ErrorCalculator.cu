/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <math.h>

namespace mgard_cuda {

template <typename T>
T L_inf_norm(size_t n, T * data) {
  T L_inf = 0;
  for (int i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
    if (temp > L_inf)
      L_inf = temp;
  }
  return L_inf;
}

template <typename T>
T L_2_norm(size_t n, T * data) {
  T L_2 = 0;
  for (int i = 0; i < n; ++i) {
    T temp = fabs(data[i]);
      L_2 += temp * temp;
  }
  return std::sqrt(L_2);
}


template <typename T>
T L_inf_error(size_t n, T * original_data, T * decompressed_data) {
  T error_L_inf_norm = 0;
  for (int i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    if (temp > error_L_inf_norm)
      error_L_inf_norm = temp;
  }
  return error_L_inf_norm;
}

template <typename T>
T L_2_error(size_t n, T * original_data, T * decompressed_data) {
  T error_L_2_norm = 0;
  for (int i = 0; i < n; ++i) {
    T temp = fabs(original_data[i] - decompressed_data[i]);
    error_L_2_norm += temp * temp;
  }
  return std::sqrt(error_L_2_norm);
}

template float L_inf_norm<float>(size_t n, float * data);
template double L_inf_norm<double>(size_t n, double * data);
template float L_2_norm<float>(size_t n, float * data);
template double L_2_norm<double>(size_t n, double * data);

template float L_inf_error<float>(size_t n, float * original_data, float * decompressed_data);
template double L_inf_error<double>(size_t n, double * original_data, double * decompressed_data);
template float L_2_error<float>(size_t n, float * original_data, float * decompressed_data);
template double L_2_error<double>(size_t n, double * original_data, double * decompressed_data);
}