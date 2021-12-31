cmake_minimum_required(VERSION 3.4...3.18)
set(CMAKE_CXX_STANDARD 14)

# ref : https://stackoverflow.com/questions/50028830/pybind11-where-to-put-pybind11-module
set(PYBIND11_PYTHON_VERSION 2.7)
add_subdirectory(../pybind11 pybind11_2)
pybind11_add_module(unet_cpp_extension2 SHARED ../src/unet_code.cpp)
set_target_properties(unet_cpp_extension2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{HOME}/.local/lib/python2.7/site-packages/)
target_compile_definitions(unet_cpp_extension2 PUBLIC -DPYBIND=2)

find_package(OpenCV 3 REQUIRED )
find_package(PCL 1. REQUIRED)
#message("!!!!!!!!!!!!!!!!!!!!")
#message(${PCL_INCLUDE_DIRS})
target_include_directories(unet_cpp_extension2 PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(unet_cpp_extension2 PUBLIC ${OpenCV_LIBS} ${PCL_LIBRARIES})

