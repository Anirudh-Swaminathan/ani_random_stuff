############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################
cmake_minimum_required(VERSION 3.5)
# collect source files
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#project(hw1_ani LANGUAGES C CXX CUDA)
project(hw1_ani)

# anicodebreaker lines
include(FindCUDA)
find_package(CUDA REQUIRED)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp reference_calc.cpp compare.cpp)

# CUDA_ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
cuda_add_executable(HW1 ${HW1_files} ${hdr} ${cu})
