cmake_minimum_required(VERSION 3.4...3.18)
project(py2ext)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3 REQUIRED )
find_package(PCL 1. REQUIRED COMPONENTS common filters)
find_package(PCL 1. REQUIRED COMPONENTS segmentation) # It cause error for python3, but it works for python2. ???

set(PYBIND11_PYTHON_VERSION 2.7)
add_subdirectory(../pybind11 pybind11_2)
pybind11_add_module(unet_ext2 SHARED ../src/unet_code.cpp)
set_target_properties(unet_ext2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{HOME}/.local/lib/python2.7/site-packages/)
target_include_directories(unet_ext2 PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(unet_ext2 PUBLIC ${OpenCV_LIBS} ${PCL_LIBRARIES})
set_target_properties(unet_ext2 PROPERTIES OUTPUT_NAME "unet_ext")

