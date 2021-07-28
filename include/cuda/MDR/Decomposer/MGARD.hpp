#ifndef _MDR_MGARD_DECOMPOSER_HPP
#define _MDR_MGARD_DECOMPOSER_HPP

#include "DecomposerInterface.hpp"
// #include "decompose.hpp"
// #include "recompose.hpp"
#include "../../DataRefactoring.h"
#include <cstring>

namespace MDR {
    // MGARD decomposer with orthogonal basis
    template<class T>
    class MGARDOrthoganalDecomposer : public concepts::DecomposerInterface<T> {
    public:
        MGARDOrthoganalDecomposer(){}
        void decompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            // MGARD::Decomposer<T> decomposer;
            // std::vector<size_t> dims(dimensions.size());
            // for(int i=0; i<dims.size(); i++){
            //     dims[i] = dimensions[i];
            // }
            // decomposer.decompose(data, dims, target_level);
            if (dimensions.size() == 1) {
                size_t size = 1;
                std::vector<mgard_cuda::SIZE> shape(dimensions.size());
                for(int i=0; i<shape.size(); i++){
                    shape[i] = dimensions[i];
                    size *= dimensions[i];
                }
                mgard_cuda::Handle<1, T> handle(shape);
                mgard_cuda::Array<1, T> array(shape);
                array.loadData((const T*)data);
                handle.allocate_workspace();
                mgard_cuda::decompose<1, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
                target_level, 0);
                handle.sync_all();
                handle.free_workspace();
                std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
            } else if (dimensions.size() == 2) {
                size_t size = 1;
                std::vector<mgard_cuda::SIZE> shape(dimensions.size());
                for(int i=0; i<shape.size(); i++){
                    shape[i] = dimensions[i];
                    size *= dimensions[i];
                }
                mgard_cuda::Handle<2, T> handle(shape);
                mgard_cuda::Array<2, T> array(shape);
                array.loadData((const T*)data);
                handle.allocate_workspace();
                mgard_cuda::decompose<2, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
                target_level, 0);
                handle.sync_all();
                handle.free_workspace();
                std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
            } else if (dimensions.size() == 3) {
                printf("start decompose\n");
                size_t size = 1;
                std::vector<mgard_cuda::SIZE> shape(dimensions.size());
                for(int i=0; i<shape.size(); i++){
                    shape[i] = dimensions[i];
                    size *= dimensions[i];
                }
                mgard_cuda::Handle<3, T> handle(shape);
                mgard_cuda::Array<3, T> array(shape);
                array.loadData((const T*)data);
                handle.allocate_workspace();
                mgard_cuda::decompose<3, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
                handle.l_target, 0);
                handle.sync_all();
                handle.free_workspace();
                std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
                printf("done decompose\n");
            }

        }
        void recompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            // MGARD::Recomposer<T> recomposer;
            // std::vector<size_t> dims(dimensions.size());
            // for(int i=0; i<dims.size(); i++){
            //     dims[i] = dimensions[i];
            // }
            // recomposer.recompose(data, dims, target_level);
            if (dimensions.size() == 1) {
                size_t size = 1;
                std::vector<mgard_cuda::SIZE> shape(dimensions.size());
                for(int i=0; i<shape.size(); i++){
                    shape[i] = dimensions[i];
                    size *= dimensions[i];
                }
                mgard_cuda::Handle<1, T> handle(shape);
                mgard_cuda::Array<1, T> array(shape);
                array.loadData((const T*)data);
                handle.allocate_workspace();
                mgard_cuda::recompose<1, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
                target_level, 0);
                handle.sync_all();
                handle.free_workspace();
                std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
            } else if (dimensions.size() == 2) {
                size_t size = 1;
                std::vector<mgard_cuda::SIZE> shape(dimensions.size());
                for(int i=0; i<shape.size(); i++){
                    shape[i] = dimensions[i];
                    size *= dimensions[i];
                }
                mgard_cuda::Handle<2, T> handle(shape);
                mgard_cuda::Array<2, T> array(shape);
                array.loadData((const T*)data);
                handle.allocate_workspace();
                mgard_cuda::recompose<2, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
                target_level, 0);
                handle.sync_all();
                handle.free_workspace();
                std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
            } else if (dimensions.size() == 3) {
                size_t size = 1;
                std::vector<mgard_cuda::SIZE> shape(dimensions.size());
                for(int i=0; i<shape.size(); i++){
                    shape[i] = dimensions[i];
                    size *= dimensions[i];
                }
                mgard_cuda::Handle<3, T> handle(shape);
                mgard_cuda::Array<3, T> array(shape);
                array.loadData((const T*)data);
                handle.allocate_workspace();
                mgard_cuda::recompose<3, T>(handle, array.get_dv(), array.get_ldvs_h(), array.get_ldvs_d(),
                target_level, 0);
                handle.sync_all();
                handle.free_workspace();
                std::memcpy(data, array.getDataHost(), size*sizeof(T)); 
            }
        }
        void print() const {
            std::cout << "MGARD orthogonal decomposer" << std::endl;
        }
    };
    // MGARD decomposer with hierarchical basis
    template<class T>
    class MGARDHierarchicalDecomposer : public concepts::DecomposerInterface<T> {
    public:
        MGARDHierarchicalDecomposer(){}
        void decompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            // MGARD::Decomposer<T> decomposer;
            // std::vector<size_t> dims(dimensions.size());
            // for(int i=0; i<dims.size(); i++){
            //     dims[i] = dimensions[i];
            // }
            // decomposer.decompose(data, dims, target_level, true);
        }
        void recompose(T * data, const std::vector<uint32_t>& dimensions, uint32_t target_level) const {
            // MGARD::Recomposer<T> recomposer;
            // std::vector<size_t> dims(dimensions.size());
            // for(int i=0; i<dims.size(); i++){
            //     dims[i] = dimensions[i];
            // }
            // recomposer.recompose(data, dims, target_level, true);
        }
        void print() const {
            std::cout << "MGARD hierarchical decomposer" << std::endl;
        }
    };
}
#endif
