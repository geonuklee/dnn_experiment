#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

#include <utils.h>
#include <segment2d.h>

namespace py = pybind11;

template<typename T>
cv::Mat GetCvMat(long rows, long cols, long dim);

template<>
cv::Mat GetCvMat<unsigned char>(long rows, long cols, long dim){
  if(dim == 3)
    return cv::Mat(rows, cols, CV_8UC3);
  return cv::Mat(rows, cols, CV_8UC1);
}

template<>
cv::Mat GetCvMat<float>(long rows, long cols, long dim){
  if(dim == 3)
    return cv::Mat(rows, cols, CV_32FC3);
  return cv::Mat(rows, cols, CV_32FC1);
}

template<>
cv::Mat GetCvMat<int32_t>(long rows, long cols, long dim){
  if(dim == 3)
    return cv::Mat(rows, cols, CV_32SC3);
  return cv::Mat(rows, cols, CV_32SC1);
}

template<typename T>
cv::Mat array2cvMat(py::array_t<T> array){
  py::buffer_info buf_info = array.request();
  long rows = buf_info.shape[0];
  long cols = buf_info.shape[1];
  const T* ptr = (const T*) buf_info.ptr;
  long dim = buf_info.shape.size()==2 ? 1 : buf_info.shape[2];
  cv::Mat mat = GetCvMat<T>(rows, cols, dim);
  memcpy(mat.data, ptr, rows*cols*dim * sizeof(T));
  return mat;
}

template<typename T>
py::array_t<T> cvMat2array(cv::Mat mat){
  assert(sizeof(T) == mat.elemSize());
  py::array_t<T> array = py::array_t<T>(mat.total() * mat.elemSize() );
  py::buffer_info buf_info = array.request();
  T* ptr = (T*) buf_info.ptr;
  memcpy(ptr, mat.data, mat.total()*mat.elemSize() );
  if(mat.channels() > 1)
    array.resize({(long)mat.rows, (long)mat.cols, (long)mat.channels() });
  else
    array.resize({(long)mat.rows, (long)mat.cols});
  return array;
}

class PySegment2DEdgeBased {
public:
  PySegment2DEdgeBased(const std::string& name)
    : segment2d_( std::make_shared<Segment2DEdgeBased>(name) )
  {
  }

  py::array_t<int32_t> Process(py::array_t<unsigned char> _rgb,
               py::array_t<float> _depth,
               py::array_t<unsigned char> _outline_edge,
               py::array_t<unsigned char> _convex_edge,
               float fx,
               float fy
              ) {
    cv::Mat rgb = array2cvMat(_rgb);
    cv::Mat depth = array2cvMat(_depth);
    cv::Mat outline_edge = array2cvMat(_outline_edge);
    cv::Mat convex_edge = array2cvMat(_convex_edge);

    std::map<int,int> ins2cls;
    bool verbose = false;
    cv::Mat surebox; // None surebox
    segment2d_->SetEdge(outline_edge, convex_edge, surebox);

    cv::Mat marker;
    bool b = segment2d_->Process(rgb,depth, marker, convex_edge, ins2cls, verbose);
    assert(b); // TODO Deal with it.

    py::array_t<int32_t> _marker = cvMat2array<int32_t>(marker);
    return _marker;
  }
  
protected:
  std::shared_ptr<Segment2DEdgeBased> segment2d_;
};

PYBIND11_MODULE(unetsegment, m) {
  py::class_<PySegment2DEdgeBased>(m, "Segment2DEdgeBased")
      .def(py::init<const std::string &>())
      .def("Process", &PySegment2DEdgeBased::Process, "Segment instance from given edges",
           py::arg("rgb"),
           py::arg("depth"),
           py::arg("outline_edge"),
           py::arg("convex_edge"),
           py::arg("fx"), py::arg("fy")
           );
}

