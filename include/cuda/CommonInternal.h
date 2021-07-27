/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#ifndef MGRAD_CUDA_COMMON_INTERNAL
#define MGRAD_CUDA_COMMON_INTERNAL

#include <stdint.h>

#define MAX_GRID_X 2147483647
#define MAX_GRID_Y 65536
#define MAX_GRID_Z 65536

#define COPY 0
#define ADD 1
#define SUBTRACT 2

#define SIGNATURE_SIZE 16
#define SIGNATURE "MGARD_CUDA_V010"

#include <algorithm>
#include <cstdio>


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Common.h"




// #define WARP_SIZE 32
// #define ROUND_UP_WARP(TID) ((TID) + WARP_SIZE - 1) / WARP_SIZE




#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

namespace mgard_cuda {

template <typename T> struct quant_meta {
  char signature[SIGNATURE_SIZE] = SIGNATURE;
  SIZE l_target;
  bool gpu_lossless;
  bool enable_lz4;
  SIZE dict_size;
  T norm;
  T tol;
  T s;
};

template <DIM D> int check_shape(std::vector<SIZE> shape);

bool is_2kplus1_cuda(double num);

// __device__ int get_idx(const int ld, const int i, const int j);

// __device__ int get_idx(const int ld1, const int ld2, const int i, const int
// j,
//                        const int k);

// __forceinline__ __device__ int get_idx(const int ld, const int i, const int j) {
//   return ld * i + j;
// }

// ld2 = nrow
// ld1 = pitch
// for 1-3D
__host__ __forceinline__ __device__ LENGTH
get_idx(const SIZE ld1, const SIZE ld2, const SIZE z, const SIZE y, const SIZE x) {
  return ld2 * ld1 * z + ld1 * y + x;
}

// for 3D+
__host__ __forceinline__ __device__ LENGTH
get_idx(const LENGTH ld1, const LENGTH ld2, const SIZE z, const SIZE y, const SIZE x) {
  return ld2 * ld1 * z + ld1 * y + x;
}

// leading dimension first
__host__ inline LENGTH get_idx(std::vector<SIZE> lds, std::vector<SIZE> idx) {
  LENGTH curr_stride = 1;
  LENGTH ret_idx = 0;
  for (DIM i = 0; i < idx.size(); i++) {
    ret_idx += idx[i] * curr_stride;
    curr_stride *= lds[i];
  }
  return ret_idx;
}

template <DIM D> __forceinline__ __device__ LENGTH get_idx(SIZE *lds, SIZE *idx) {
  LENGTH curr_stride = 1;
  LENGTH ret_idx = 0;
  for (DIM i = 0; i < D; i++) {
    ret_idx += idx[i] * curr_stride;
    curr_stride *= lds[i];
  }
  return ret_idx;
}

__host__ inline std::vector<SIZE> gen_idx(DIM D, DIM curr_dim_r, DIM curr_dim_c,
                                         DIM curr_dim_f, SIZE idx_r, SIZE idx_c,
                                         SIZE idx_f) {
  std::vector<SIZE> idx(D, 0);
  idx[curr_dim_r] = idx_r;
  idx[curr_dim_c] = idx_c;
  idx[curr_dim_f] = idx_f;
  return idx;
}

__host__ __forceinline__ __device__ int div_roundup(SIZE a, SIZE b) {
  return (a - 1) / b + 1;
}

// template <int D, int R, int C, int F>
// __host__ inline void kernel_config(thrust::device_vector<int> &shape, int &tbx,
//                                    int &tby, int &tbz, int &gridx, int &gridy,
//                                    int &gridz,
//                                    thrust::device_vector<int> &assigned_dimx,
//                                    thrust::device_vector<int> &assigned_dimy,
//                                    thrust::device_vector<int> &assigned_dimz) {

//   tbx = F;
//   tby = C;
//   tbz = R;
//   gridx = ceil((double)shape[0] / F);
//   gridy = ceil((double)shape[1] / C);
//   gridz = ceil((double)shape[2] / R);
//   assigned_dimx.push_back(0);
//   assigned_dimy.push_back(1);
//   assigned_dimz.push_back(2);

//   int d = 3;
//   while (d < D) {
//     if (gridx * shape[d] < MAX_GRID_X) {
//       gridx *= shape[d];
//       assigned_dimx.push_back(d);
//       d++;
//     } else {
//       break;
//     }
//   }

//   while (d < D) {
//     if (gridy * shape[d] < MAX_GRID_Y) {
//       gridy *= shape[d];
//       assigned_dimy.push_back(d);
//       d++;
//     } else {
//       break;
//     }
//   }

//   while (d < D) {
//     if (gridz * shape[d] < MAX_GRID_Z) {
//       gridz *= shape[d];
//       assigned_dimz.push_back(d);
//       d++;
//     } else {
//       break;
//     }
//   }
// }

// template <int D, int R, int C, int F>
// __forceinline__ __device__ void
// get_idx(int *shape, int assigned_nx, int *assigned_dimx, int assigned_ny,
//         int *assigned_dimy, int assigned_nz, int *assigned_dimz, int *idx) {
//   int bidx = blockIdx.x;
//   int bidy = blockIdx.y;
//   int bidz = blockIdx.z;
//   idx[0] = (bidx % shape[0]) * F + threadIdx.x;
//   idx[1] = (bidy % shape[1]) * C + threadIdx.y;
//   idx[2] = (bidz % shape[2]) * R + threadIdx.z;
//   if (idx[0] < 0) {
//     printf("neg %d %d %d %d\n", bidx, shape[0], F, threadIdx.x);
//   }
//   if (idx[1] < 0) {
//     printf("neg %d %d %d %d\n", bidy, shape[1], C, threadIdx.y);
//   }
//   if (idx[2] < 0) {
//     printf("neg %d %d %d %d\n", bidz, shape[2], R, threadIdx.z);
//   }
//   // bidx /= shape[0];
//   // bidy /= shape[1];
//   // bidz /= shape[2];
//   // for (int i = 1; i < assigned_nx; i++) {
//   //   int d = assigned_dimx[i];
//   //   idx[d] = bidx%shape[d];
//   //   bidx /= shape[d];
//   // }
//   // for (int i = 1; i < assigned_ny; i++) {
//   //   int d = assigned_dimy[i];
//   //   idx[d] = bidy%shape[d];
//   //   bidy /= shape[d];
//   // }
//   // for (int i = 1; i < assigned_nz; i++) {
//   //   int d = assigned_dimz[i];
//   //   idx[d] = bidz%shape[d];
//   //   bidz /= shape[d];
//   // }
// }

template <typename T> T max_norm_cuda(const T *v, size_t size);

template <typename T> __device__ T _get_dist(T *coords, int i, int j);

// __host__ __device__ int get_lindex_cuda(const int n, const int no, const int i);

template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template <typename T> __device__ inline T lerp(T v0, T v1, T t) {
  // return fma(t, v1, fma(-t, v0, v0));
#ifdef MGARD_CUDA_FMA
  if (sizeof(T) == sizeof(double)) {
    return fma(t, v1, fma(-t, v0, v0));
  } else if (sizeof(T) == sizeof(float)) {
    return fmaf(t, v1, fmaf(-t, v0, v0));
  }
#else
  T r = v0 + v0 * t * -1;
  r = r + t * v1;
  return r;
#endif
}

template <typename T>
__device__ inline T mass_trans(T a, T b, T c, T d, T e, T h1, T h2, T h3, T h4,
                               T r1, T r2, T r3, T r4) {
  T tb, tc, td, tb1, tb2, tc1, tc2, td1, td2;
// #ifdef MGARD_CUDA_FMA
//   if (sizeof(T) == sizeof(double)) {
//     tb1 = fma(c, h2, a * h1);
//     tb2 = fma(b, h2, b * h1);

//     tc1 = fma(d, h3, b * h2);
//     tc2 = fma(c, h3, c * h2);

//     td1 = fma(c, h4, e * h3);
//     td2 = fma(d, h4, d * h3);

//     tb = fma(2, tb2, tb1);
//     tc = fma(2, tc2, tc1);
//     td = fma(2, td2, td1);
//     return fma(td, r4, fma(tb, r1, tc));
//   } else if (sizeof(T) == sizeof(float)) {
//     tb1 = fmaf(c, h2, a * h1);
//     tb2 = fmaf(b, h2, b * h1);

//     tc1 = fmaf(d, h3, b * h2);
//     tc2 = fmaf(c, h3, c * h2);

//     td1 = fmaf(c, h4, e * h3);
//     td2 = fmaf(d, h4, d * h3);

//     tb = fmaf(2, tb2, tb1);
//     tc = fmaf(2, tc2, tc1);
//     td = fmaf(2, td2, td1);
//     return fmaf(td, r4, fmaf(tb, r1, tc));
//   }
// #else

  if (h1 + h2 != 0) {
    r1 = h1 / (h1 + h2);
  } else {
    r1 = 0.0;
  }
  if (h3 + h4 != 0) {
    r4 = h4 / (h3 + h4);
  } else {
    r4 = 0.0;
  }

  // printf("%f %f %f %f %f (%f %f %f %f)\n", a, b, c, d, e, h1, h2, h3, h4);
  tb = a * (h1/6) + b * ((h1 + h2)/3) + c * (h2/6);
  tc = b * (h2/6) + c * ((h2 + h3)/3) + d * (h3/6);
  td = c * (h3/6) + d * ((h3 + h4)/3) + e * (h4/6);
  tc += tb * r1 + td * r4;
  return tc;
// #endif
}

template <typename T>
__device__ inline T tridiag_forward(T prev, T bm, T curr) {

#ifdef MGARD_CUDA_FMA
  if (sizeof(T) == sizeof(double)) {
    return fma(prev, bm, curr);
  } else if (sizeof(T) == sizeof(float)) {
    return fmaf(prev, bm, curr);
  }
#else
  return curr - prev * bm;
#endif
}

template <typename T>
__device__ inline T tridiag_backward(T prev, T dist, T am, T curr) {

#ifdef MGARD_CUDA_FMA
  if (sizeof(T) == sizeof(double)) {
    return fma(-1 * dist, prev, curr) * am;
  } else if (sizeof(T) == sizeof(float)) {
    return fmaf(-1 * dist, prev, curr) * am;
  }
#else
  return (curr - dist * prev) / am;
#endif
}


template <typename T>
__device__ inline T tridiag_forward2(T prev, T am, T bm, T curr) {

// #ifdef MGARD_CUDA_FMA
//   if (sizeof(T) == sizeof(double)) {
//     return fma(prev, bm, curr);
//   } else if (sizeof(T) == sizeof(float)) {
//     return fmaf(prev, bm, curr);
//   }
// #else
  // printf("am: %f bm: %f\n", am, bm);
  return curr - prev * (am / bm);
// #endif
}

template <typename T>
__device__ inline T tridiag_backward2(T prev, T am, T bm, T curr) {

// #ifdef MGARD_CUDA_FMA
//   if (sizeof(T) == sizeof(double)) {
//     return fma(-1 * dist, prev, curr) * am;
//   } else if (sizeof(T) == sizeof(float)) {
//     return fmaf(-1 * dist, prev, curr) * am;
//   }
// #else
  // printf("am: %f bm: %f\n", am, bm);
  return (curr - am * prev) / bm;
// #endif
}

} // namespace mgard_cuda

#endif