#ifndef _MDR_PERBIT_BP_ENCODER_GPU_HPP
#define _MDR_PERBIT_BP_ENCODER_GPU_HPP

#include "../../CommonInternal.h"
#include "../../Functor.h"
#include "../../AutoTuner.h"
#include "../../Task.h"
#include "../../DeviceAdapters/DeviceAdapterCuda.h"

#include "BitplaneEncoderInterface.hpp"
#include <bitset>
namespace mgard_cuda {
namespace MDR {
  template <DIM D, typename T, typename T_fp, SIZE B>
    class PER_BIT_ENCODER_Functor: public Functor<D, T> {
      public: 
      MGARDm_CONT PER_BIT_ENCODER_Functor(SIZE n,
                                          SIZE num_bitplanes,
                                          T exp,
                                          SubArray<1, T> v,
                                          SubArray<1, T> level_errors):
                                          n(n), num_bitplanes(num_bitplanes),
                                          exp(exp),
                                          v(v), level_errors(level_errors) {
                                            Functor<D, T>();
                                            num_bitplanes_roundup_warp = 
                                              (num_bitplanes/MGARDm_WARP_SIZE)*MGARDm_WARP_SIZE;
                                          }
      
      // exponent align
      // calculate error
      // store signs
      // find the most significant bit
      MGARDm_EXEC void
      __operation1(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) 
      {
        // assume 1D parallelization
        // B needs to be a multiply of MGARDm_WARP_SIZE
        // B >= num_bitplanes

        int8_t * sm_p = (int8_t *)shared_memory;
        sm_level_errors = (T*)sm_p; sm_p += ((num_bitplanes + 1) * B) * sizeof(T);
        sm_fix_point = (T_fp*)sm_p; sm_p += B * sizeof(T_fp);
        sm_bit_planes = (uint64_t*)sm_p; sm_p += num_bitplanes * (((B * 2)-1)/sizeof(uint64_t)+1) * sizeof(uint64_t);
        sm_first_bit_pos = (SIZE*)sm_p; sm_p += B * sizeof(SIZE);
        sm_signs = (SIZE*)sm_p; sm_p += B * sizeof(SIZE);
        sm_pos = (SIZE*)sm_p; sm_p += num_bitplanes_roundup_warp * sizeof(SIZE);



        local_idx = threadx;
        global_idx = blockx * nblockx + local_idx;
        
        // memset sm_level_errors to 0 to avoid thread divergence when reduce
        for(int k=0; k<num_bitplanes+1; k++) {
          sm_level_errors[k * B + local_idx] = 0;
        }
        // memset sm_pos to 0 
        if (local_idx < num_bitplanes_roundup_warp) sm_pos[local_idx] = 0;

        if (global_idx < n) {
          T cur_data = *v(global_idx);
          T shifted_data = ldexp(cur_data, num_bitplanes - exp);
          sm_signs[local_idx] = signbit(cur_data); //not using if to void thread divergence
          int64_t fix_point = (int64_t) shifted_data;
          T_fp fp_data = fabs(fix_point);

          
          //collect errors 
          T abs_shifted_data = fabs(shifted_data);
          T mantissa = abs_shifted_data - fp_data;
          sm_level_errors[num_bitplanes * B + local_idx] =
            mantissa * mantissa;
          for(int k=1; k<num_bitplanes; k++){
              uint64_t mask = (1 << k) - 1;
              T diff = (T) (fp_data & mask) + mantissa;
              sm_level_errors[(num_bitplanes - k) * B + local_idx] = 
                diff * diff;
          }
          sm_level_errors[local_idx] += abs_shifted_data * abs_shifted_data;

          // detect first bit per elems
          // { // option 1: iteratively detect most significant bit
          //   bool first_bit_detected = false;
          //   for(int k=num_bitplanes-1; k>=0; k--){ 
          //     uint8_t index = num_bitplanes - 1 - k;
          //     uint8_t bit = (fp_data >> k) & 1u;
          //     // printf("%f %f %d\n", fp_data, fp_data>>k, (fp_data >> k) & 1u);
          //     if (bit && !first_bit_detected) {
          //       sm_first_bit_pos[local_idx] = index;
          //       first_bit_detected = true;
          //       printf("first_bit_detected: %d\n", index);
          //     }
          //   }
          // }

          { // option 2: reverse fixpoint & use __ffsll detect least significant bit
            printf("sizeof(T_fp) = %u\n", sizeof(T_fp));
            if (sizeof(T_fp) == sizeof(uint32_t)) {
              sm_first_bit_pos[local_idx] = __ffs(__brev(fp_data)) - (32 - num_bitplanes) - 1;
            } else if (sizeof(T_fp) == sizeof(uint64_t)) {
              sm_first_bit_pos[local_idx] = __ffsll(__brevll(fp_data)) - (64 - num_bitplanes) - 1;
            }
          }

          // save fp_data to shared memory
          sm_fix_point[local_idx] = fp_data;
          printf("gpu fp_data[%llu] %u, sign: %u, first_bit: %u\n", global_idx, fp_data, sm_signs[local_idx], sm_first_bit_pos[local_idx]);

        }
      }

      // convert fix point to bit-planes
      // level error reduction (wrap level)
      MGARDm_EXEC void
      __operation2(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) {

        { //option 1: one bit-plane per thread + put sign after most significant bit
          if (local_idx < num_bitplanes) { // each thread reponsibles for one bit-plane
            SIZE pos_in_buffer = 0;
            SIZE pos = 0;
            uint64_t buffer = 0;
            for (SIZE i = 0; i < B; i++) {
              SIZE k = local_idx;
              uint8_t index = num_bitplanes - 1 - k;
              uint8_t bit = (sm_fix_point[i] >> k) & 1u;
              buffer += bit << pos_in_buffer;
              pos_in_buffer ++;
              if (pos_in_buffer == 64) {
                sm_bit_planes[pos*num_bitplanes+index] = buffer;
                pos++;
                buffer = 0;
                pos_in_buffer = 0;
              }
              if (index == sm_first_bit_pos[i]) {
                buffer += sm_signs[i] << pos_in_buffer;
                pos_in_buffer ++;
                if (pos_in_buffer == 64) {
                  sm_bit_planes[pos*num_bitplanes+index] = buffer;
                  pos++;
                  buffer = 0;
                  pos_in_buffer = 0;
                }
              }
            }
            sm_pos[local_idx] = pos;
          }
        }

        { //option 2: one bit-plane per thread + store signs as a seperate bit-plane

        }


        { // level error reduction (intra warp)
          int lane = threadIdx.x % MGARDm_WARP_SIZE;
          int wid = threadIdx.x / MGARDm_WARP_SIZE;
          if (global_idx < n) {
            for (int i = 0; i < num_bitplanes+1; i++) {
              T error = sm_level_errors[i * B + local_idx];
              for (int offset = MGARDm_WARP_SIZE/2; offset > 0; offset /= 2) {
                error += __shfl_down_sync(0xffffffff, error, offset);
              }
              if (lane == 0) {
                sm_level_errors[i * B + local_idx] = error;
              }
            }
          }
        }        
      }

      // get max bit-plane length (intra warp)
      // level error reduction (inter warp)
      MGARDm_EXEC void
      __operation3(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) {

        { // get max bit-plane length (intra warp)
          int lane = threadIdx.x % MGARDm_WARP_SIZE;
          int wid = threadIdx.x / MGARDm_WARP_SIZE;
          if (local_idx < num_bitplanes_roundup_warp) {
            SIZE bitplane_length = sm_pos[local_idx];
            for (int offset = MGARDm_WARP_SIZE/2; offset > 0; offset /= 2) {
              bitplane_length = max(bitplane_length, __shfl_down_sync(0xffffffff, bitplane_length, offset));
            }
            if (lane == 0) {
              sm_pos[local_idx] = bitplane_length;
            }
          }
        }

        { // level error reduction (inter warp)
          SIZE num_active_threads = B/MGARDm_WARP_SIZE;
          unsigned int mask = 0;
          for (SIZE i = 0; i < num_active_threads; i++) {
            mask += 1u << i;
          }
          int lane = threadIdx.x % MGARDm_WARP_SIZE;
          int wid = threadIdx.x / MGARDm_WARP_SIZE;

          if (local_idx < num_active_threads) {
            for (int i = 0; i < num_bitplanes+1; i++) {
              T error = sm_level_errors[i * B + local_idx * MGARDm_WARP_SIZE];
              for (int offset = MGARDm_WARP_SIZE/2; offset > 0; offset /= 2) {
                error += __shfl_down_sync(mask, error, offset);
              }
              if (lane == 0) {
                sm_level_errors[i * B] = error;
              }
            }
          }
        }
      }

      MGARDm_EXEC void
      __operation4(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) {
        { // get max bit-plane length (inter warp)
          SIZE num_active_threads = num_bitplanes_roundup_warp/MGARDm_WARP_SIZE;
          unsigned int mask = 0;
          for (SIZE i = 0; i < num_active_threads; i++) {
            mask += 1u << i;
          }
          int lane = threadIdx.x % MGARDm_WARP_SIZE;
          int wid = threadIdx.x / MGARDm_WARP_SIZE;
          
          if (local_idx < num_active_threads) {
            SIZE bitplane_length = sm_pos[local_idx * MGARDm_WARP_SIZE];
            for (int offset = MGARDm_WARP_SIZE/2; offset > 0; offset /= 2) {
              bitplane_length = max(bitplane_length, __shfl_down_sync(mask, bitplane_length, offset));
            }
            if (lane == 0) {
              sm_pos[0] = bitplane_length;
            }
          }
        }

      }

      MGARDm_EXEC void
      __operation5(IDX ngridz, IDX ngridy, IDX ngridx,
               IDX nblockz, IDX nblocky, IDX nblockx,
               IDX blockz, IDX blocky, IDX blockx,
               IDX threadz, IDX thready, IDX threadx, T * shared_memory) {

      }
    private:
      // parameters
      SIZE n;
      SIZE num_bitplanes;
      T exp;
      SubArray<1, T> v;
      SubArray<1, T> level_errors;

      // stateful thread local variables
      IDX local_idx, global_idx;
      SIZE num_bitplanes_roundup_warp;
      T * sm_level_errors;
      T_fp * sm_fix_point;
      SIZE * sm_first_bit_pos;
      uint64_t * sm_bit_planes;
      SIZE * sm_signs;
      SIZE * sm_pos;

  };

  template <DIM D, typename T, typename DEVICE>
  class PER_BIT_ENCODER_AutoTuner: public AutoTuner<D, T, DEVICE> {
  public:
    MGARDm_CONT
    PER_BIT_ENCODER_AutoTuner(Handle<D, T> &handle):AutoTuner<D, T, DEVICE>(handle) {}

    template <typename T_fp, SIZE B>
    MGARDm_CONT
    Task<D, T, PER_BIT_ENCODER_Functor<D, T, T_fp, B> > GenTask(SIZE n,
                                                             SIZE num_bitplanes,
                                                             T exp,
                                                             SubArray<1, T> v,
                                                             SubArray<1, T> level_errors,
                                                             int queue_idx) 
    {
      using FunctorType = PER_BIT_ENCODER_Functor<D, T, T_fp, B>;
      FunctorType functor(n, num_bitplanes, exp, v, level_errors);
        SIZE total_thread_z = 1;
        SIZE total_thread_y = 1;
        SIZE total_thread_x = n;
        SIZE tbx, tby, tbz, gridx, gridy, gridz;
        size_t sm_size = 0;
        tbz = 1;
        tby = 1;
        tbx = B;
        sm_size += (B * (num_bitplanes + 1)) * sizeof(T);
        sm_size += B * sizeof(T_fp);
        sm_size += (((B * 2)-1)/sizeof(uint64_t)+1) * num_bitplanes;
        sm_size += B * sizeof(SIZE);
        sm_size += B * sizeof(SIZE);
        gridz = ceil((float)total_thread_z / tbz);
        gridy = ceil((float)total_thread_y / tby);
        gridx = ceil((float)total_thread_x / tbx);
        return Task<D, T, FunctorType>(functor, gridz, gridy, gridx, 
                          tbz, tby, tbx, sm_size, queue_idx); 
    }

    MGARDm_CONT
    void Execute(SIZE n,
                 SIZE num_bitplanes,
                 T exp,
                 SubArray<1, T> v,
                 SubArray<1, T> level_errors,
                 int queue_idx) {
      
      if (std::is_same<T, double>::value) {
        using FunctorType = PER_BIT_ENCODER_Functor<D, T, uint64_t, 32>;
        using TaskType = Task<D, T, FunctorType>;
        TaskType task = GenTask<uint64_t, 32>(n, num_bitplanes, exp, v, level_errors, queue_idx);
        DeviceAdapter<D, T, TaskType, DEVICE> adapter(this->handle);
        adapter.Execute(task);
      } else if (std::is_same<T, float>::value) {
        using FunctorType = PER_BIT_ENCODER_Functor<D, T, uint32_t, 32>;
        using TaskType = Task<D, T, FunctorType>;
        TaskType task = GenTask<uint32_t, 32>(n, num_bitplanes, exp, v, level_errors, queue_idx);
        DeviceAdapter<D, T, TaskType, DEVICE> adapter(this->handle);
        adapter.Execute(task);
      }
    }
  };

    class BitEncoderGPU
    {
    public:
        BitEncoderGPU(uint64_t * stream_begin_pos){
            stream_begin = stream_begin_pos;
            stream_pos = stream_begin;
            buffer = 0;
            position = 0;
        }
        void encode(uint64_t b){
            buffer += b << position;
            position ++;
            if(position == 64){
                // printf("encoder buffer full\n");
                *(stream_pos ++) = buffer;
                buffer = 0;
                position = 0;
            }
        }
        void flush(){
            if(position){
                *(stream_pos ++) = buffer;
                buffer = 0;
                position = 0;
            }
        }
        uint32_t size(){
            return (stream_pos - stream_begin);
        }
    private:
        uint64_t buffer = 0;
        uint8_t position = 0;
        uint64_t * stream_pos = NULL;
        uint64_t * stream_begin = NULL;
    };

    class BitDecoderGPU{
    public:
        BitDecoderGPU(uint64_t const * stream_begin_pos){
            stream_begin = stream_begin_pos;
            stream_pos = stream_begin;
            buffer = 0;
            position = 0;
        }
        uint8_t decode(){
            if(position == 0){
                buffer = *(stream_pos ++);
                position = 64;
            }
            uint8_t b = buffer & 1u;
            buffer >>= 1;
            position --;
            return b;
        }
        uint32_t size(){
            return (stream_pos - stream_begin);
        }
    private:
        uint64_t buffer = 0;
        uint8_t position = 0;
        uint64_t const * stream_pos = NULL;
        uint64_t const * stream_begin = NULL;
    };

    #define PER_BIT_BLOCK_SIZE 1
    // per bit bitplane encoder that encodes data by bit using T_stream type buffer
    template<DIM D, typename T_data, typename T_stream>
    class PerBitBPEncoderGPU : public concepts::BitplaneEncoderInterface<D, T_data> {
    public:
        PerBitBPEncoderGPU(Handle<D, T_data> &handle): _handle(handle) {
            std::cout <<  "PerBitBPEncoder\n";
            static_assert(std::is_floating_point<T_data>::value, "PerBitBPEncoderGPU: input data must be floating points.");
            static_assert(!std::is_same<T_data, long double>::value, "PerBitBPEncoderGPU: long double is not supported.");
            static_assert(std::is_unsigned<T_stream>::value, "PerBitBPEncoderGPU: streams must be unsigned integers.");
            static_assert(std::is_integral<T_stream>::value, "PerBitBPEncoderGPU: streams must be unsigned integers.");
        }


        std::vector<uint8_t *> encode(T_data const * data, int32_t n, int32_t exp, uint8_t num_bitplanes, std::vector<uint32_t>& stream_sizes) const {
            
            assert(num_bitplanes > 0);
            // determine block size based on bitplane integer type
            const int32_t block_size = PER_BIT_BLOCK_SIZE;
            stream_sizes = std::vector<uint32_t>(num_bitplanes, 0);
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            std::vector<uint8_t *> streams;
            for(int i=0; i<num_bitplanes; i++){
                streams.push_back((uint8_t *) malloc(2 * n / UINT8_BITS + sizeof(uint64_t)));
            }
            std::vector<BitEncoderGPU> encoders;
            for(int i=0; i<streams.size(); i++){
                encoders.push_back(BitEncoderGPU(reinterpret_cast<uint64_t*>(streams[i])));
            }
            T_data const * data_pos = data;

            for(int i=0; i<n - block_size; i+=block_size){
                T_stream sign_bitplane = 0;
                for(int j=0; j<block_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    bool sign = cur_data < 0;
                    int64_t fix_point = (int64_t) shifted_data;
                    T_fp fp_data = sign ? -fix_point : +fix_point;
                    // compute level errors
                    bool first_bit = true;
                    for(int k=num_bitplanes - 1; k>=0; k--){
                        uint8_t index = num_bitplanes - 1 - k;
                        uint8_t bit = (fp_data >> k) & 1u;
                        encoders[index].encode(bit);
                        if(bit && first_bit){
                            encoders[index].encode(sign);
                            first_bit = false;
                        }
                    }                    
                }
            }
            // leftover
            {
                int rest_size = n % block_size;
                if(rest_size == 0) rest_size = block_size;
                for(int j=0; j<rest_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    bool sign = cur_data < 0;
                    int64_t fix_point = (int64_t) shifted_data;
                    T_fp fp_data = sign ? -fix_point : +fix_point;
                    // compute level errors
                    bool first_bit = true;
                    for(int k=num_bitplanes - 1; k>=0; k--){
                        uint8_t index = num_bitplanes - 1 - k;
                        uint8_t bit = (fp_data >> k) & 1u;
                        encoders[index].encode(bit);
                        if(bit && first_bit){
                            encoders[index].encode(sign);
                            first_bit = false;
                        }
                    }                    
                }
            }
            for(int i=0; i<num_bitplanes; i++){
                encoders[i].flush();
                stream_sizes[i] = encoders[i].size() * sizeof(uint64_t);
            }
            return streams;
        }

        // only differs in error collection
        std::vector<uint8_t *> encode(T_data const * data, int32_t n, int32_t exp, uint8_t num_bitplanes, std::vector<uint32_t>& stream_sizes, std::vector<double>& level_errors) const {
            
            Array<1, T_data> v_array({(SIZE)n});
            v_array.loadData(data);
            SubArray<1, T_data> v(v_array);
            Array<1, T_data> level_errors_array({(SIZE)num_bitplanes});
            SubArray<1, T_data> level_errors_subarray(level_errors_array);
            PER_BIT_ENCODER_AutoTuner<D, T_data, CUDA>(_handle).Execute(n, num_bitplanes, exp, v, level_errors_subarray, 0);

            assert(num_bitplanes > 0);
            // determine block size based on bitplane integer type
            const int32_t block_size = PER_BIT_BLOCK_SIZE;
            stream_sizes = std::vector<uint32_t>(num_bitplanes, 0);
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            std::vector<uint8_t *> streams;
            for(int i=0; i<num_bitplanes; i++){
                streams.push_back((uint8_t *) malloc(2 * n / UINT8_BITS + sizeof(uint64_t)));
            }
            std::vector<BitEncoderGPU> encoders;
            for(int i=0; i<streams.size(); i++){
                encoders.push_back(BitEncoderGPU(reinterpret_cast<uint64_t*>(streams[i])));
            }
            // init level errors
            level_errors.clear();
            level_errors.resize(num_bitplanes + 1);
            for(int i=0; i<level_errors.size(); i++){
                level_errors[i] = 0;
            }
            T_data const * data_pos = data;
            // printf("n = %u\n",n);
            for(int i=0; i<n - block_size; i+=block_size){
                T_stream sign_bitplane = 0;
                for(int j=0; j<block_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    bool sign = cur_data < 0;
                    int64_t fix_point = (int64_t) shifted_data;
                    T_fp fp_data = sign ? -fix_point : +fix_point;
                    printf("cpu fp_data[%d] %llu\n", data_pos-data-1, fp_data);
                    // compute level errors
                    collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
                    bool first_bit = true;
                    for(int k=num_bitplanes - 1; k>=0; k--){
                        uint8_t index = num_bitplanes - 1 - k;
                        uint8_t bit = (fp_data >> k) & 1u;
                        encoders[index].encode(bit);
                        // printf("encode bitplane[%u] <- %u from %u\n", index, bit, data_pos-data);
                        if(bit && first_bit){
                            printf("first bit: %hu\n", index);
                            encoders[index].encode(sign);
                            first_bit = false;
                        }
                    }                    
                }
            }
            // leftover
            {
                int rest_size = n % block_size;
                if(rest_size == 0) rest_size = block_size;
                for(int j=0; j<rest_size; j++){
                    T_data cur_data = *(data_pos++);
                    T_data shifted_data = ldexp(cur_data, num_bitplanes - exp);
                    bool sign = cur_data < 0;
                    int64_t fix_point = (int64_t) shifted_data;
                    T_fp fp_data = sign ? -fix_point : +fix_point;
                    printf("cpu fp_data[%d] %llu\n", data_pos-data-1, fp_data);
                    // compute level errors
                    collect_level_errors(level_errors, fabs(shifted_data), num_bitplanes);
                    bool first_bit = true;
                    for(int k=num_bitplanes - 1; k>=0; k--){
                        uint8_t index = num_bitplanes - 1 - k;
                        uint8_t bit = (fp_data >> k) & 1u;
                        encoders[index].encode(bit);
                        // printf("encode bitplane[%u] <- %u from %u\n", index, bit, data_pos-data);
                        if(bit && first_bit){
                            // printf("encode sign bitplane[%u] <- from %u\n", index, data_pos-data);
                            encoders[index].encode(sign);
                            first_bit = false;
                        }
                    }                    
                }
            }
            for(int i=0; i<num_bitplanes; i++){
                encoders[i].flush();
                stream_sizes[i] = encoders[i].size() * sizeof(uint64_t);
                // printf("stream_sizes[%d]: %llu\n", i, stream_sizes[i]);
            }
            // translate level errors
            for(int i=0; i<level_errors.size(); i++){
                level_errors[i] = ldexp(level_errors[i], 2*(- num_bitplanes + exp));
            }
            return streams;
        }

        T_data * decode(const std::vector<uint8_t const *>& streams, int32_t n, int exp, uint8_t num_bitplanes) {
            const int32_t block_size = PER_BIT_BLOCK_SIZE;
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            T_data * data = (T_data *) malloc(n * sizeof(T_data));
            if(num_bitplanes == 0){
                memset(data, 0, n * sizeof(T_data));
                return data;
            }
            std::vector<BitDecoderGPU> decoders;
            for(int i=0; i<streams.size(); i++){
                decoders.push_back(BitDecoderGPU(reinterpret_cast<uint64_t const*>(streams[i])));
                decoders[i].size();
            }
            // decode
            T_data * data_pos = data;
            for(int i=0; i<n - block_size; i+=block_size){
                for(int j=0; j<block_size; j++){
                    T_fp fp_data = 0;
                    // decode each bit of the data for each level component
                    bool first_bit = true;
                    bool sign = false;
                    for(int k=num_bitplanes - 1; k>=0; k--){
                        uint8_t index = num_bitplanes - 1 - k;
                        uint8_t bit = decoders[index].decode();
                        fp_data += bit << k;
                        if(bit && first_bit){
                            // decode sign
                            sign = decoders[index].decode();
                            first_bit = false;
                        }
                    }
                    T_data cur_data = ldexp((T_data)fp_data, - num_bitplanes + exp);
                    *(data_pos++) = sign ? -cur_data : cur_data;
                }
            }
            // leftover
            {
                int rest_size = n % block_size;
                if(rest_size == 0) rest_size = block_size;
                for(int j=0; j<rest_size; j++){
                    T_fp fp_data = 0;
                    // decode each bit of the data for each level component
                    bool first_bit = true;
                    bool sign = false;
                    for(int k=num_bitplanes - 1; k>=0; k--){
                        uint8_t index = num_bitplanes - 1 - k;
                        uint8_t bit = decoders[index].decode();
                        fp_data += bit << k;
                        if(bit && first_bit){
                            // decode sign
                            sign = decoders[index].decode();
                            first_bit = false;
                        }
                    }
                    T_data cur_data = ldexp((T_data)fp_data, - num_bitplanes + exp);
                    *(data_pos++) = sign ? -cur_data : cur_data;
                }
            }
            return data;
        }

        T_data * progressive_decode(const std::vector<uint8_t const *>& streams, int32_t n, int exp, uint8_t starting_bitplane, uint8_t num_bitplanes, int level) {
            const int32_t block_size = PER_BIT_BLOCK_SIZE;
            // define fixed point type
            using T_fp = typename std::conditional<std::is_same<T_data, double>::value, uint64_t, uint32_t>::type;
            T_data * data = (T_data *) malloc(n * sizeof(T_data));
            if(num_bitplanes == 0){
                memset(data, 0, n * sizeof(T_data));
                return data;
            }
            std::vector<BitDecoderGPU> decoders;
            for(int i=0; i<streams.size(); i++){
                decoders.push_back(BitDecoderGPU(reinterpret_cast<uint64_t const*>(streams[i])));
                decoders[i].size();
            }
            if(level_signs.size() == level){
                level_signs.push_back(std::vector<bool>(n, false));
                sign_flags.push_back(std::vector<bool>(n, false));
            }
            std::vector<bool>& signs = level_signs[level];
            std::vector<bool>& flags = sign_flags[level];
            const uint8_t ending_bitplane = starting_bitplane + num_bitplanes;
            // decode
            T_data * data_pos = data;
            for(int i=0; i<n - block_size; i+=block_size){
                for(int j=0; j<block_size; j++){
                    T_fp fp_data = 0;
                    // decode each bit of the data for each level component
                    bool sign = false;
                    if(flags[i + j]){
                        // sign recorded
                        sign = signs[i + j];
                        for(int k=num_bitplanes - 1; k>=0; k--){
                            uint8_t index = num_bitplanes - 1 - k;
                            uint8_t bit = decoders[index].decode();
                            fp_data += bit << k;
                        }
                    }
                    else{
                        // decode sign if possible
                        bool first_bit = true;
                        for(int k=num_bitplanes - 1; k>=0; k--){
                            uint8_t index = num_bitplanes - 1 - k;
                            uint8_t bit = decoders[index].decode();
                            fp_data += bit << k;
                            if(bit && first_bit){
                                // decode sign
                                sign = decoders[index].decode();
                                first_bit = false;
                                flags[i + j] = true;
                            }
                        }
                        signs[i + j] = sign;
                    }
                    T_data cur_data = ldexp((T_data)fp_data, - ending_bitplane + exp);
                    *(data_pos++) = sign ? -cur_data : cur_data;
                }
            }
            // leftover
            {
                int rest_size = n % block_size;
                if(rest_size == 0) rest_size = block_size;
                for(int j=0; j<rest_size; j++){
                    T_fp fp_data = 0;
                    // decode each bit of the data for each level component
                    bool sign = false;
                    if(flags[n - rest_size + j]){
                        sign = signs[n - rest_size + j];
                        for(int k=num_bitplanes - 1; k>=0; k--){
                            uint8_t index = num_bitplanes - 1 - k;
                            uint8_t bit = decoders[index].decode();
                            fp_data += bit << k;
                        }
                    }
                    else{
                        bool first_bit = true;
                        for(int k=num_bitplanes - 1; k>=0; k--){
                            uint8_t index = num_bitplanes - 1 - k;
                            uint8_t bit = decoders[index].decode();
                            fp_data += bit << k;
                            if(bit && first_bit){
                                // decode sign
                                sign = decoders[index].decode();
                                first_bit = false;
                                flags[n - rest_size + j] = true;
                            }
                        }
                        signs[n - rest_size + j] = sign;
                    }
                    T_data cur_data = ldexp((T_data)fp_data, - ending_bitplane + exp);
                    *(data_pos++) = sign ? -cur_data : cur_data;
                }
            }
            return data;
        }
        void print() const {
            std::cout << "Per-bit bitplane encoder" << std::endl;
        }
    private:
        inline void collect_level_errors(std::vector<double>& level_errors, float data, int num_bitplanes) const {
            uint32_t fp_data = (uint32_t) data;
            double mantissa = data - (uint32_t) data;
            level_errors[num_bitplanes] += mantissa * mantissa;
            for(int k=1; k<num_bitplanes; k++){
                uint32_t mask = (1 << k) - 1;
                double diff = (double) (fp_data & mask) + mantissa;
                level_errors[num_bitplanes - k] += diff * diff;
            }
            level_errors[0] += data * data;
        }
        Handle<D, T_data> &_handle;
        std::vector<std::vector<bool>> level_signs;
        std::vector<std::vector<bool>> sign_flags;
    };
}
}
#endif
