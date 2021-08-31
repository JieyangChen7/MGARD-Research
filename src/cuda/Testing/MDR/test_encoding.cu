#include "cuda/CommonInternal.h"
#include "cuda/Functor.h"
#include "cuda/AutoTuner.h"
#include "cuda/Task.h"
#include "cuda/DeviceAdapters/DeviceAdapterCuda.h"
#include <mma.h>
using namespace nvcuda;

#include <chrono>
using namespace std::chrono;
template <typename T>
MGARDm_CONT_EXEC void
print_bits(T v, int num_bits, bool reverse = false) {
  for (int j = 0; j < num_bits; j++) {
    if (!reverse) printf("%u", (v >> sizeof(T)*8-1-j) & 1u);
    else printf("%u", (v >> j) & 1u);
  }
}


namespace mgard_cuda {

// #define ALIGN_LEFT 0 // for encoding
// #define ALIGN_RIGHT 1 // for decoding

// typedef unsigned long long int uint64_cu;

// template <typename T_org, typename T_trans, SIZE nblockx, SIZE nblocky, SIZE nblockz, SIZE ALIGN, SIZE METHOD> 
// struct BlockBitTranspose<T_org, T_trans, nblockx, nblocky, nblockz, ALIGN, METHOD, CUDA> {
  
//   typedef cub::WarpReduce<T_trans> WarpReduceType;
//   using WarpReduceStorageType = typename WarpReduceType::TempStorage;

//   typedef cub::BlockReduce<T_trans, nblockx, cub::BLOCK_REDUCE_WARP_REDUCTIONS, nblocky, nblockz> BlockReduceType;
//   using BlockReduceStorageType = typename BlockReduceType::TempStorage;

//   MGARDm_EXEC 
//   void Serial_All(T_org * v, T_trans * tv, SIZE b, SIZE B) {
//     if (threadIdx.x == 0 && threadIdx.y == 0) {
//       for (SIZE B_idx = 0; B_idx < B; B_idx++) {
//         T_trans buffer = 0; 
//         for (SIZE b_idx = 0; b_idx < b; b_idx++) {
//           T_trans bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
//           if (ALIGN == ALIGN_LEFT) {
//             buffer += bit << (sizeof(T_trans)*8-1-b_idx); 
//             // if (B_idx == 0) {
//             //   printf("%u %u %u\n", B_idx, b_idx, bit);
//             //   print_bits(buffer, sizeof(T_trans)*8, false);
//             //   printf("\n");
//             // }
//           } else if (ALIGN == ALIGN_RIGHT) {
//             buffer += bit << (b-1-b_idx); 
//             // if (b_idx == 0) printf("%u %u %u\n", B_idx, b_idx, bit);
//           } else { }
//           // if (j == 0 ) {printf("i %u j %u shift %u bit %u\n", i,j,b-1-j, bit); }
//         }

//         // printf("buffer[%u]: %llu\n", B_idx, buffer);

//         tv[B_idx] = buffer;
//       }
//     }
//   }

//   MGARDm_EXEC 
//   void Parallel_B_Serial_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {
//     if (threadIdx.y == 0) {
//       for (SIZE B_idx = threadIdx.x; B_idx < B; B_idx += 32) {
//         T_trans buffer = 0; 
//         for (SIZE b_idx = 0; b_idx < b; b_idx++) {
//           T_trans bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
//           if (ALIGN == ALIGN_LEFT) {
//             buffer += bit << sizeof(T_trans)*8-1-b_idx; 
//           } else if (ALIGN == ALIGN_RIGHT) {
//             buffer += bit << (b-1-b_idx); 
//             // if (b_idx == 0) printf("%u %u %u\n", B_idx, b_idx, bit);
//           } else { }
//         }
//         tv[B_idx] = buffer;
//       }
//     }
//   }  

//   MGARDm_EXEC 
//   void Parallel_B_Atomic_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {
//     if (threadIdx.x < b && threadIdx.y < B) {
//       SIZE i = threadIdx.x;
//       for (SIZE B_idx = threadIdx.y; B_idx < B; B_idx += 32) {
//         for (SIZE b_idx = threadIdx.x; b_idx < b; b_idx += 32) {
//           T_trans bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
//           T_trans shifted_bit;
//           if (ALIGN == ALIGN_LEFT) {
//             shifted_bit = bit << sizeof(T_trans)*8-1-b_idx;
//           } else if (ALIGN == ALIGN_RIGHT) {
//             shifted_bit = bit << b-1-b_idx;
//           } else { }
//           T_trans * sum = &(tv[B_idx]);
//           // atomicAdd(sum, shifted_bit);
//         }
//       }
//     }
//   }  

//   MGARDm_EXEC 
//   void Parallel_B_Reduction_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {

//     // __syncthreads(); long long start = clock64();

//     __shared__ WarpReduceStorageType warp_storage[32];

//     SIZE warp_idx = threadIdx.y;
//     SIZE lane_idx = threadIdx.x;
//     SIZE B_idx, b_idx;
//     T_trans bit = 0;
//     T_trans shifted_bit = 0;
//     T_trans sum = 0;

//     // __syncthreads(); start = clock64() - start;
//     // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
//     // __syncthreads(); start = clock64();

//     for (SIZE B_idx = threadIdx.y; B_idx < B; B_idx += 32) {
//       sum = 0;
//       for (SIZE b_idx = threadIdx.x; b_idx < ((b-1)/32+1)*32; b_idx += 32) {
//         shifted_bit = 0;
//         if (b_idx < b && B_idx < B) {
//           bit = (v[b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
//           if (ALIGN == ALIGN_LEFT) {
//             shifted_bit = bit << sizeof(T_trans)*8-1-b_idx; 
//           } else if (ALIGN == ALIGN_RIGHT) {
//             shifted_bit = bit << b-1-b_idx; 
//           } else { }
//         }

//         // __syncthreads(); start = clock64() - start;
//         // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);
//         // __syncthreads(); start = clock64();

//         sum += WarpReduceType(warp_storage[warp_idx]).Sum(shifted_bit);
//         // if (B_idx == 32) printf("shifted_bit[%u] %u sum %u\n", b_idx, shifted_bit, sum);
//       }
//       if (lane_idx == 0) {
//         tv[B_idx] = sum;
//       }
//     }

//     // __syncthreads(); start = clock64() - start;
//     // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);
//     // __syncthreads(); start = clock64();
//   }  

//   MGARDm_EXEC 
//   void Parallel_B_Ballot_b(T_org * v, T_trans * tv, SIZE b, SIZE B) {

//     SIZE warp_idx = threadIdx.y;
//     SIZE lane_idx = threadIdx.x;
//     SIZE B_idx, b_idx;
//     int bit = 0;
//     T_trans sum = 0;
    
//     // __syncthreads(); start = clock64() - start;
//     // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
//     // __syncthreads(); start = clock64();


//     for (SIZE B_idx = threadIdx.y; B_idx < B; B_idx += 32) {
//       sum = 0;
//       SIZE shift = 0;
//       for (SIZE b_idx = threadIdx.x; b_idx < ((b-1)/32+1)*32; b_idx += 32) {
//         bit = 0;
//         if (b_idx < b && B_idx < B) {
//           if (ALIGN == ALIGN_LEFT) {
//             bit = (v[sizeof(T_trans)*8-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
//           } else if (ALIGN == ALIGN_RIGHT) {
//             bit = (v[b-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
//           } else { }
//         }

//         // __syncthreads(); start = clock64() - start;
//         // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);
//         // __syncthreads(); start = clock64();
//         sum += ((T_trans)__ballot_sync (0xffffffff, bit)) << shift;
//         // sum += WarpReduceType(warp_storage[warp_idx]).Sum(shifted_bit);
//         // if (B_idx == 32) printf("shifted_bit[%u] %u sum %u\n", b_idx, shifted_bit, sum);
//         shift += 32;
//       }
//       if (lane_idx == 0) {
//         tv[B_idx] = sum;
//       }
//     }

//     // __syncthreads(); start = clock64() - start;
//     // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);
//     // __syncthreads(); start = clock64();

//     // __syncthreads(); long long start = clock64();

//     // SIZE i = threadIdx.x;
//     // SIZE B_idx = threadIdx.y;
//     // SIZE b_idx = threadIdx.x;

//     // __syncthreads(); start = clock64() - start;
//     // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);
//     // __syncthreads(); start = clock64();


//     // int bit = (v[b-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;

//     // __syncthreads();
//     // start = clock64() - start;
//     // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);
//     // __syncthreads(); start = clock64();

//     // printf("b_idx[%u]: bit %d\n", b_idx, bit);
//     // unsigned int sum = __ballot_sync (0xffffffff, bit);
//     // printf("b_idx[%u]: sum %u\n", b_idx, sum);
//     // if (b_idx) tv[B_idx] = sum;

//     // __syncthreads(); start = clock64() - start;
//     // if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);
//     // __syncthreads(); start = clock64();
//   }

//   MGARDm_EXEC 
//   void TCU(T_org * v, T_trans * tv, SIZE b, SIZE B) {
//     __syncthreads();
//     long long start = clock64();

//     __shared__ half tile_a[16*16];
//     __shared__ half tile_b[32*32];
//     __shared__ float output[32*32];
//     uint8_t bit;
//     half shifted_bit;
//     SIZE i = threadIdx.x;
//     SIZE B_idx = threadIdx.y;
//     SIZE b_idx = threadIdx.x;
//     SIZE warp_idx = threadIdx.y;
//     SIZE lane_idx = threadIdx.x;
    
//     __syncthreads();
//     start = clock64() - start;
//     if (threadIdx.y == 0 && threadIdx.x == 0) printf("time0: %llu\n", start);

//     __syncthreads();
//     start = clock64();
//     __syncthreads();
   
//     if (threadIdx.x < B * b) {
//       uint8_t bit = (v[sizeof(T_trans)*8-1-b_idx] >> (sizeof(T_org)*8 - 1 - B_idx)) & 1u;
//       shifted_bit = bit << (sizeof(T_trans)*8-1-b_idx) % 8; 
//       tile_b[b_idx * 32 + B_idx] = shifted_bit;
//       if (i < 8) { 
//         tile_a[i] = 1u;
//         tile_a[i+8] = 1u << 8;
//       }
//     }
//     __syncthreads();
//     start = clock64() - start;
//     if (threadIdx.y == 0 && threadIdx.x == 0) printf("time1: %llu\n", start);

//     __syncthreads();
//     start = clock64();
//     __syncthreads();
    
//     if (warp_idx < 4) { 
//       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
//       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
//       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
//       wmma::load_matrix_sync(a_frag, tile_a, 16);
//       wmma::load_matrix_sync(b_frag, tile_b + (warp_idx/2)*16 + (warp_idx%2)*16, 32);
//       wmma::fill_fragment(c_frag, 0.0f);
//       wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
//       wmma::store_matrix_sync(output+ (warp_idx/2)*16 + (warp_idx%2)*16, c_frag, 32, wmma::mem_row_major);
//     }

//     __syncthreads();
//     start = clock64() - start;
//     if (threadIdx.y == 0 && threadIdx.x == 0) printf("time2: %llu\n", start);

//   }

//   MGARDm_EXEC 
//   void Transpose(T_org * v, T_trans * tv, SIZE b, SIZE B) {
//     if (METHOD == 0)  Serial_All(v, tv, b, B);
//     else if (METHOD == 1) Parallel_B_Serial_b(v, tv, b, B);
//     else if (METHOD == 2) Parallel_B_Atomic_b(v, tv, b, B);
//     else if (METHOD == 3) Parallel_B_Reduction_b(v, tv, b, B);
//     else if (METHOD == 4) Parallel_B_Ballot_b(v, tv, b, B);
//     // else if (METHOD == 5) TCU(v, tv, b, B);
//   }
// };


template <typename T_org, typename T_trans, SIZE METHOD, typename DeviceType>
class EncodingTestFunctor : public Functor<DeviceType> {
public: 
  EncodingTestFunctor(SubArray<1, T_org> v,
                      SubArray<1, T_trans> bitplane,
                      SIZE num_bitplanes): v(v), bitplane(bitplane), 
                              num_bitplanes(num_bitplanes) {
    Functor<DeviceType>();
  }
  MGARDm_EXEC void
  Operation1() {

    local_data_idx = this->thready * this->nblockx + this->threadx;
    global_data_idx = this->blockx * num_elems_per_TB + local_data_idx;

    local_bitplane_idx = this->thready * this->nblockx + this->threadx;
    global_bitplane_idx = this->blockx * sizeof(T_org)*8 + local_data_idx;

    int8_t * sm_p = (int8_t *)this->shared_memory;
    sm_v        = (T_org*)  sm_p; sm_p += num_elems_per_TB * sizeof(T_org);
    sm_bitplane = (T_trans*)sm_p; sm_p += sizeof(T_org)*8 * sizeof(T_trans);

    if (local_data_idx < num_elems_per_TB) {
      sm_v[local_data_idx] = *v(global_data_idx);
      // printf("sm_v[%d]: %u\n", local_data_idx, sm_v[local_data_idx]);
    }

    if (local_bitplane_idx < num_bitplanes) {
      sm_bitplane[local_bitplane_idx] = 0;
    }
  }

  MGARDm_EXEC void
  Operation2() {
    // __syncthreads();
    // long long start = clock64();
    blockBitTranspose.Transpose(sm_v, sm_bitplane, num_elems_per_TB, num_bitplanes);
    // __syncthreads();
    // start = clock64() - start;

    // if (this->blockz == 0 && this->blocky == 0 && this->blockx == 0 &&
    //         this->threadx == 0 && this->thready == 0 && this->threadz == 0) 
      // printf("time: %llu\n", start);

  }

  MGARDm_EXEC void
  Operation3() {
    if (local_data_idx < num_bitplanes) {
      *bitplane(global_bitplane_idx) = sm_bitplane[local_data_idx];
      // printf("tv[%u]: %u\n", global_bitplane_idx, sm_bitplane[local_data_idx]);
    }  
  }

  MGARDm_EXEC void
  Operation4() {}

  MGARDm_EXEC void
  Operation5() {}


  MGARDm_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size += num_elems_per_TB * sizeof(T_org);
    size += sizeof(T_org)*8 * sizeof(T_trans);
    return size;
  }

private:
  SubArray<1, T_org> v;
  SubArray<1, T_trans> bitplane;
  SIZE num_bitplanes;
  const SIZE num_elems_per_TB = sizeof(T_trans) * 8;
  const SIZE thread_per_TB = sizeof(T_org)*8 * sizeof(T_trans)*8;
  SIZE local_data_idx, global_data_idx;
  SIZE local_bitplane_idx, global_bitplane_idx;
  T_org   * sm_v;
  T_trans * sm_bitplane;  
  BlockBitTranspose<T_org, T_trans, sizeof(T_org)*8 * sizeof(T_trans)*8, 1, 1, ALIGN_LEFT, METHOD, DeviceType> blockBitTranspose;
};



template <typename HandleType, typename T_org, typename T_trans, SIZE METHOD, typename DeviceType>
  class EncodingTest: public AutoTuner<HandleType, DeviceType> {
  public:
    MGARDm_CONT
    EncodingTest(HandleType& handle):AutoTuner<HandleType, DeviceType>(handle) {}

    MGARDm_CONT
    Task<EncodingTestFunctor<T_org, T_trans, METHOD, DeviceType> > GenTask(SIZE n,
                                                                  SubArray<1, T_org> v,
                                                                  SubArray<1, T_trans> bitplane,
                                                                  SIZE num_bitplanes,
                                                                  int queue_idx) 
    {
      using FunctorType = EncodingTestFunctor<T_org, T_trans, METHOD, DeviceType>;
      FunctorType functor(v, bitplane, num_bitplanes);
        SIZE total_thread_z = 1;
        SIZE total_thread_y = 1;
        SIZE total_thread_x = n ;
        SIZE tbx, tby, tbz, gridx, gridy, gridz;
        size_t sm_size = functor.shared_memory_size();
        tbz = 1;
        tby = 32;
        tbx = 32;
        gridz = 1;//ceil((float)total_thread_z / tbz);
        gridy = 1;//ceil((float)total_thread_y / tby);
        gridx = (n/(sizeof(T_trans)*8));ceil((float)total_thread_x / tbx);
        printf("EncodingTest config(%u %u %u) (%u %u %u)\n", tbx, tby, tbz, gridx, gridy, gridz);
        return Task(functor, gridz, gridy, gridx, 
                          tbz, tby, tbx, sm_size, queue_idx); 
    }

    MGARDm_CONT
    void Execute(SIZE n,
                  SubArray<1, T_org> v,
                  SubArray<1, T_trans> bitplane,
                  SIZE num_bitplanes,
                  int queue_idx) {
      using FunctorType = EncodingTestFunctor<T_org, T_trans, METHOD, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task = GenTask(n, v, bitplane, num_bitplanes, queue_idx);
      DeviceAdapter<HandleType, TaskType, DeviceType> adapter(this->handle);
      adapter.Execute(task);
    }
  };


template <typename T_trans, typename T_org, SIZE METHOD, typename DeviceType>
class DecodingTestFunctor : public Functor<DeviceType> {
public: 
  DecodingTestFunctor(SubArray<1, T_trans> bitplane,
                      SubArray<1, T_org> v,
                      SIZE num_bitplanes): v(v), bitplane(bitplane), 
                              num_bitplanes(num_bitplanes) {
    Functor<DeviceType>();
  }
  MGARDm_EXEC void
  Operation1() {

    local_data_idx = this->thready * this->nblockx + this->threadx;
    global_data_idx = this->blockx * num_elems_per_TB + local_data_idx;

    local_bitplane_idx = this->thready * this->nblockx + this->threadx;
    global_bitplane_idx = this->blockx * sizeof(T_org)*8 + local_bitplane_idx;

    int8_t * sm_p = (int8_t *)this->shared_memory;
    sm_bitplane = (T_trans*)sm_p; sm_p += sizeof(T_org)*8 * sizeof(T_trans);
    sm_v        = (T_org*)  sm_p; sm_p += num_elems_per_TB * sizeof(T_org);
    
    if (local_data_idx < num_elems_per_TB) {
      sm_v[local_data_idx] = 0;
      // printf("sm_v[%d]: %u\n", local_data_idx, sm_v[local_data_idx]);
    }

    if (local_bitplane_idx < num_bitplanes) {
      sm_bitplane[local_bitplane_idx] = *bitplane(global_bitplane_idx);
      // printf("sm_bitplane[%d]: %u\n", local_bitplane_idx, sm_bitplane[local_bitplane_idx]);
    }
  }

  MGARDm_EXEC void
  Operation2() {
    // blockBitTranspose.Init();
    // __syncthreads();
    // long long start = clock64();
    blockBitTranspose.Transpose(sm_bitplane, sm_v, num_bitplanes, sizeof(T_trans)*8);
    // __syncthreads();
    // start = clock64() - start;

    // if (this->blockz == 0 && this->blocky == 0 && this->blockx == 0 &&
    //         this->threadx == 0 && this->thready == 0 && this->threadz == 0) 
    //   printf("time: %llu\n", start);

  }

  MGARDm_EXEC void
  Operation3() {
    if (local_data_idx < num_elems_per_TB) {
      *v(global_data_idx) = sm_v[local_data_idx];
      // printf("tv[%u]: %u\n", global_bitplane_idx, sm_bitplane[local_data_idx]);
    }  
  }

  MGARDm_EXEC void
  Operation4() {}

  MGARDm_EXEC void
  Operation5() {}


  MGARDm_CONT size_t
  shared_memory_size() {
    size_t size = 0;
    size += num_elems_per_TB * sizeof(T_org);
    size += sizeof(T_org)*8 * sizeof(T_trans);
    return size;
  }

private:
  SubArray<1, T_org> v;
  SubArray<1, T_trans> bitplane;
  SIZE num_bitplanes;
  const SIZE num_elems_per_TB = sizeof(T_trans) * 8;
  const SIZE thread_per_TB = sizeof(T_org)*8 * sizeof(T_trans)*8;
  SIZE local_data_idx, global_data_idx;
  SIZE local_bitplane_idx, global_bitplane_idx;
  T_org   * sm_v;
  T_trans * sm_bitplane;  
  BlockBitTranspose<T_trans, T_org, sizeof(T_org)*8 * sizeof(T_trans)*8, 1, 1, ALIGN_RIGHT, METHOD, DeviceType> blockBitTranspose;

};



template <typename HandleType, typename T_trans, typename T_org, SIZE METHOD, typename DeviceType>
  class DecodingTest: public AutoTuner<HandleType, DeviceType> {
  public:
    MGARDm_CONT
    DecodingTest(HandleType& handle):AutoTuner<HandleType, DeviceType>(handle) {}

    MGARDm_CONT
    Task<DecodingTestFunctor<T_trans, T_org, METHOD, DeviceType> > GenTask(SIZE n,
                                                                  SubArray<1, T_trans> bitplane,
                                                                  SubArray<1, T_org> v,
                                                                  SIZE num_bitplanes,
                                                                  int queue_idx) 
    {
      using FunctorType = DecodingTestFunctor<T_trans, T_org, METHOD, DeviceType>;
      FunctorType functor(bitplane, v, num_bitplanes);
        // SIZE total_thread_z = 1;
        // SIZE total_thread_y = 1;
        // SIZE total_thread_x = n ;
        SIZE tbx, tby, tbz, gridx, gridy, gridz;
        size_t sm_size = functor.shared_memory_size();
        tbz = 1;
        tby = 32;
        tbx = 32;
        gridz = 1;//ceil((float)total_thread_z / tbz);
        gridy = 1;//ceil((float)total_thread_y / tby);
        gridx = n/(sizeof(T_trans)*8);//ceil((float)total_thread_x / tbx);
        // printf("DecodingTest config(%u %u %u) (%u %u %u)\n", tbx, tby, tbz, gridx, gridy, gridz);
        return Task(functor, gridz, gridy, gridx, 
                          tbz, tby, tbx, sm_size, queue_idx); 
    }

    MGARDm_CONT
    void Execute(SIZE n,
                  SubArray<1, T_trans> bitplane,
                  SubArray<1, T_org> v,
                  SIZE num_bitplanes,
                  int queue_idx) {
      using FunctorType = DecodingTestFunctor<T_trans, T_org, METHOD, DeviceType>;
      using TaskType = Task<FunctorType>;
      TaskType task = GenTask(n, bitplane, v, num_bitplanes, queue_idx);
      DeviceAdapter<HandleType, TaskType, DeviceType> adapter(this->handle);
      adapter.Execute(task);
    }
  };
}



template <typename T_org, typename T_trans, mgard_cuda::SIZE METHOD>
void test(size_t num_blocks, int encoding_num_bitplanes, int decoding_num_bitplanes){
  
  // using T_org = uint64_t;
  // using T_trans = uint8_t;
  // int encoding_num_bitplanes = 40;
  // int decoding_num_bitplanes = 10;
  // size_t num_blocks = 1;//30000000;

  size_t n = sizeof(T_trans) * 8 * num_blocks;
  // const mgard_cuda::SIZE METHOD = 0;

  double total_data = (n*encoding_num_bitplanes)/8/1e9;
  printf("T_org: %d, T_trans: %d, Method: %d, Total data: %.2e GB.\n", 
          sizeof(T_org)*8, sizeof(T_trans)*8, METHOD, total_data);


  high_resolution_clock::time_point t1, t2, start, end;
  duration<double> time_span;

  // printf("Generating data\n");
  T_org * v = new T_org[n];
  T_org * v2 = new T_org[n];
  for (int i = 0; i < n; i++) {
    // v[i] = rand() % 10000000;
    v2[i] = 0;
    v[i] = 0;
    for (int j = 0; j < sizeof(T_org)*8; j++) {
      T_org bit = rand() % 2;
      v[i] += bit << j;
    }
  }
  // printf("Done Generating\n");

  // printf("Original data:\n");
  // for (int i = 0; i < n; i++) {
  //   printf("[%d]%llu:\t", i, v[i]);
  //   print_bits(v[i], sizeof(T_org)*8, false);
  //   printf("\n");
  //   if ((i+1) % (sizeof(T_trans) * 8) == 0) {
  //     printf("\n");
  //   }
  // }
  // printf("\n");


  T_trans * bitplane = new T_trans[num_blocks*sizeof(T_org)*8];

  mgard_cuda::Handle<1, float> handle;

  mgard_cuda::Array<1, T_org> v_array({(mgard_cuda::SIZE)n});
  v_array.loadData(v);
  mgard_cuda::SubArray<1, T_org> v_subarray(v_array);

  mgard_cuda::Array<1, T_trans> bitplane_array({(mgard_cuda::SIZE)num_blocks*(int)sizeof(T_org)*8});
  mgard_cuda::SubArray<1, T_trans> bitplane_subarray(bitplane_array);


  mgard_cuda::Array<1, T_org> v2_array({(mgard_cuda::SIZE)n});
  mgard_cuda::SubArray<1, T_org> v2_subarray(v2_array);

  // printf("Starting encoding\n");
  handle.sync_all();
  t1 = high_resolution_clock::now();
  mgard_cuda::EncodingTest<mgard_cuda::Handle<1, float>, T_org, T_trans, METHOD, mgard_cuda::CUDA>(handle).Execute(n, v_subarray, bitplane_subarray, encoding_num_bitplanes, 0);
  handle.sync_all();
  t2 = high_resolution_clock::now();
  time_span = duration_cast<duration<double>>(t2 - t1);
  
  // printf("Done encoding\n");
  
  std::cout << "Encoding time: " << time_span.count() <<" s (" << total_data/time_span.count() << "GB/s)\n";
  T_trans * bitplanes = bitplane_array.getDataHost();
  bool pass = true;
  printf("Encoding: ");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < encoding_num_bitplanes; j++) {
      uint8_t bit1 = (v[i] >> sizeof(T_org)*8-1-j) & 1u;
      uint8_t bit2 = (bitplanes[j+i/(sizeof(T_trans)*8)*sizeof(T_org)*8] >> (sizeof(T_trans)*8-1-i%(sizeof(T_trans)*8)))& 1u;
      if (bit1 != bit2) {
        pass = false;
        // printf("\e[31m%u\e[0m", bit1);
      } else {
        // printf("\e[32m%u\e[0m", bit1);
      }
    }
    // printf("\n");
  }

  if (pass) printf("\e[32mpass\e[0m\n");
  else printf("\e[31mno pass\e[0m\n");

  

  // printf("Bitplane:\n");
  // for (int i = 0; i < num_blocks*encoding_num_bitplanes; i++) {
  //   printf("[%d]%llu:\t", i, bitplanes[i]);
  //   print_bits(bitplanes[i], sizeof(T_trans)*8, false);
  //   printf("\n");
  //   if ((i+1) % (sizeof(T_org) * 8) == 0) {
  //     printf("\n");
  //   }
  // }
  // printf("\n");

  handle.sync_all();
  mgard_cuda::DecodingTest<mgard_cuda::Handle<1, float>, T_trans, T_org, METHOD, mgard_cuda::CUDA>(handle).Execute(n, bitplane_subarray, v2_subarray, decoding_num_bitplanes, 0);
  handle.sync_all();

  v2 = v2_array.getDataHost();

  // for (int i = 0; i < n; i++) {
  //   printf("[%d]%llu:\t", i, v2[i]);
  //   print_bits(v2[i], sizeof(T_org)*8, false);
  //   printf("\n");
  //   if (i && i % (sizeof(T_trans) * 8) == 0) {
  //     printf("\n");
  //   }
  // }
  // printf("\n");

  printf("Decoding: ");
  pass = true;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < decoding_num_bitplanes; j++) {
      uint8_t bit1 = (v[i] >> sizeof(T_org)*8-1-j) & 1u;
      uint8_t bit2 = (v2[i] >> decoding_num_bitplanes-1-j) & 1u;
      if (bit1 != bit2) {
        pass = false;
      }
    }
  }

  if (pass) printf("\e[32mpass\e[0m\n");
  else printf("\e[31mno pass\e[0m\n");
}


template <mgard_cuda::SIZE METHOD>
void test_method() {
  typedef unsigned long long int uint64_t;

  test<uint32_t, uint8_t, METHOD>(1, 30, 10);
  test<uint32_t, uint16_t, METHOD>(1, 30, 10);
  test<uint32_t, uint32_t, METHOD>(1, 30, 10);
  test<uint32_t, uint64_t, METHOD>(1, 30, 10);

  test<uint64_t, uint8_t, METHOD>(1, 50, 10);
  test<uint64_t, uint16_t, METHOD>(1, 50, 10);
  test<uint64_t, uint32_t, METHOD>(1, 50, 10);
  test<uint64_t, uint64_t, METHOD>(1, 50, 10);
}

int main() {
  // test_method<0>();
  // test_method<1>();
  // test_method<2>();
  test_method<3>();
  test_method<4>();

  return 0;
}