#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int FindEdge(const unsigned char *arr_in, const std::vector<long int>& shape,
             unsigned char *arr_out) {
  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int hw = 2;

  for (int r0=0; r0<rows; r0++) {
    for (int c0=0; c0<cols; c0++) {
      const unsigned char index0 = arr_in[r0*cols+c0];
      //const unsigned char index0 = arr_in[c0*rows+r0];
      bool is_edge = false;
      for (int dr=-hw; dr<=hw; dr++) {
        const int r = r0 + dr;
        if( r < 0 || r >= rows)
          continue;
        for (int dc=-hw; dc<=hw; dc++) {
          const int c = c0 + dc;
          if( c < 0 || c >= cols)
            continue;
          const unsigned char index1 = arr_in[r*cols+c];
          //const unsigned char index1 = arr_in[c*rows+r];
          if(index0!=index1){
            is_edge = true;
            break;
          }
        }
        if(is_edge)
          break;
      }
      arr_out[r0*cols + c0] = is_edge?1:0;
    }
  }
  return 0;
}

void GetGradient(const float* depth, const std::vector<long int>& shape,
                 int offset,
                 float* grad) {
  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int size = 2*rows*cols;

  for (int i=0; i < size; i++)
    grad[i] = 0.;

  for (int r0=0; r0<rows; r0++) {
    for (int c0=0; c0<cols; c0++) {
      const float& d0 = depth[r0*cols+c0];
      float gx, gy;
      if(c0+offset < cols)
        gx = (depth[r0*cols+(c0+offset)] - d0 ) / offset;
      else
        gx = (d0 - depth[r0*cols+(c0-offset)] ) / offset;
      if(r0+offset < rows)
        gy = (depth[(r0+offset)*cols + c0] - d0 ) / offset;
      else
        gy = (d0 - depth[(r0-offset)*cols + c0] ) / offset;
      int idx0 = 2*(r0*cols + c0);
      grad[idx0 ] = gx;
      grad[idx0+1] = gy;
    }
  }
  return;
}


// ref) https://stackoverflow.com/questions/49582252/pybind-numpy-access-2d-nd-arrays
py::array_t<unsigned char> PyFindEdge(py::array_t<unsigned char> inputmask) {
  py::buffer_info buf_inputmask = inputmask.request();

  /*  allocate the buffer */
  py::array_t<unsigned char> output = py::array_t<unsigned char>(buf_inputmask.size);
  py::buffer_info buf_output = output.request();

  const unsigned char* ptr_inputmask = (const unsigned char*) buf_inputmask.ptr;
  unsigned char* ptr_output = (unsigned char*) buf_output.ptr;
  FindEdge(ptr_inputmask, buf_inputmask.shape, ptr_output);

  // reshape array to match input shape
  output.resize({buf_inputmask.shape[0], buf_inputmask.shape[1]});
  return output;
}

py::array_t<float> PyGetGradient(py::array_t<float> inputdepth, int offset) {
  py::buffer_info buf_inputdepth = inputdepth.request();

  /*  allocate the buffer */
  py::array_t<float> output = py::array_t<float>(2*buf_inputdepth.size);
  py::buffer_info buf_output = output.request();

  const float* ptr_inputdepth = (const float*) buf_inputdepth.ptr;
  float* ptr_output = (float*) buf_output.ptr;
  GetGradient(ptr_inputdepth, buf_inputdepth.shape, offset, ptr_output);

  // reshape array to match input shape
  output.resize({buf_inputdepth.shape[0], buf_inputdepth.shape[1], (long int) 2});
  return output;
}

PYBIND11_MODULE(unet_cpp_extension, m) {
  m.def("FindEdge", &PyFindEdge);
  m.def("GetGradient", &PyGetGradient);
}


