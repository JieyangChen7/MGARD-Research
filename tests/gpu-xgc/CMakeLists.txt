cmake_minimum_required(VERSION 3.19)

project(mgard_cuda_example CXX)
find_package(mgard REQUIRED)
set(ADIOS2_DIR /home/jieyang/opt/adios/lib/cmake/adios2)
find_package(ADIOS2 REQUIRED)

add_executable(test test.cpp)
target_link_libraries(test mgard::mgard adios2::adios2)
