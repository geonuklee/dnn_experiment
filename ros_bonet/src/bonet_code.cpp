#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int square(int x) {
  return x * x;
}

void normalize_pointcloud(const double* ptr_xyzrgb, int r, double* ptr_output){
  // bat_pc = (xyz, xyz, rgb, xyz)).reshape((1,-1,12)
  for(size_t i = 0; i < r; i++) {
    for(size_t j = 0; j < 3; j++)
      ptr_output[i*12 + j]     = ptr_xyzrgb[i*6+j];

    for(size_t j = 0; j < 3; j++)
      ptr_output[i*12 + j + 3] = ptr_xyzrgb[i*6+j];

    for(size_t j = 0; j < 3; j++)
      ptr_output[i*12 + j + 6] = ptr_xyzrgb[i*6+3+j];

    for(size_t j = 0; j < 3; j++)
      ptr_output[i*12 + j + 9] = ptr_xyzrgb[i*6+j];
  }
}


// ref) https://stackoverflow.com/questions/49582252/pybind-numpy-access-2d-nd-arrays
py::array_t<double> py_normalize_pointcloud(py::array_t<double> xyz_rgb) {
  py::buffer_info buf_xyzrgb = xyz_rgb.request();

  if (buf_xyzrgb.shape[1] != 6){
    throw std::runtime_error("Columns of xyz_rgb must be 6");
  }

  /*  allocate the buffer */
  py::array_t<double> output = py::array_t<double>(buf_xyzrgb.shape[0]*12);
  py::buffer_info buf_output = output.request();

  const double* ptr_xyzrgb = (const double*) buf_xyzrgb.ptr;
  double* ptr_output = (double*) buf_output.ptr;
  int r = buf_xyzrgb.shape[0];
  normalize_pointcloud(ptr_xyzrgb, r, ptr_output);

  // reshape array to match input shape
  output.resize({1,r,12});
  return output;
}

PYBIND11_MODULE(rosbonet_cpp_extension, m) {
  m.def("normalize_pointcloud", &py_normalize_pointcloud);
  m.def("square", &square);
}
