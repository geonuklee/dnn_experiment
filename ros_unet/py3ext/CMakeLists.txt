cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 14)

# ref : https://stackoverflow.com/questions/50028830/pybind11-where-to-put-pybind11-module
set(PYBIND11_PYTHON_VERSION 3.6)
add_subdirectory(../pybind11 pybind11_3)
pybind11_add_module(unet_cpp_extension3 SHARED ../src/unet_code.cpp)
set_target_properties(unet_cpp_extension3 PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{HOME}/.local/lib/python3.6/site-packages/)
target_compile_definitions(unet_cpp_extension3 PUBLIC -DPYBIND=3)

