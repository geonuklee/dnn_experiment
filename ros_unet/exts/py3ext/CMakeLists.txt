cmake_minimum_required(VERSION 3.4...3.18)
project(py3ext)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3 REQUIRED )
find_package(PCL 1. REQUIRED COMPONENTS common filters)

set(PYBIND11_PYTHON_VERSION 3.6)
add_subdirectory(../pybind11 pybind11_3)
pybind11_add_module(unet_ext3 SHARED ../../src/unet_code.cpp)
set_target_properties(unet_ext3 PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{HOME}/.local/lib/python3.6/site-packages/)
target_include_directories(unet_ext3 PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(unet_ext3 PUBLIC ${OpenCV_LIBS} ${PCL_LIBRARIES})
set_target_properties(unet_ext3 PROPERTIES OUTPUT_NAME "unet_ext")
