cmake_minimum_required(VERSION 3.4...3.18)
project(py3ext)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3 REQUIRED )
find_package(PCL 1 QUIET REQUIRED COMPONENTS common filters search sample_consensus)

# find_package(PCL 1 QUIET REQUIRED COMPONENTS segmentation)
# Above Depends on pcl::segmentation > pcl::io > vtk > python2 which cause import failure at python3
# To exclude libpython2.7, use hard coded library path. (I can't remove the libpython2.7 from the string with CMake)
if (TRUE)
  set(PCL_LIBRARIES
    /usr/lib/x86_64-linux-gnu/libpcl_common.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_filters.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_search.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_octree.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_ml.so.1.8
    /usr/lib/x86_64-linux-gnu/libpcl_features.so.1.8
    /lib/x86_64-linux-gnu/libutil.so.1
  )
else()
  target_compile_definitions(unet_ext3 PRIVATE WITHOUT_OBB)
endif()

set(PYBIND11_PYTHON_VERSION 3.6)
add_subdirectory(../pybind11 pybind11_3)
pybind11_add_module(unet_ext3 SHARED ../src/unet_code.cpp)
set_target_properties(unet_ext3 PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{HOME}/.local/lib/python3.6/site-packages/)
target_include_directories(unet_ext3 PUBLIC ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
target_link_libraries(unet_ext3 PUBLIC ${PCL_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(unet_ext3 PROPERTIES OUTPUT_NAME "unet_ext")

