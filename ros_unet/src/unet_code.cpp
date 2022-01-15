#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#if 1

#include <opencv2/opencv.hpp>
#include <memory.h>
#include <vector>
#include <pcl/filters/voxel_grid.h>

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
  output.resize({buf_inputdepth.shape[0], buf_inputdepth.shape[1], 2L});
  return output;
}

py::tuple PyUnprojectPointscloud(py::array_t<unsigned char> _rgb,
                                 py::array_t<float> _depth,
                                 py::array_t<int32_t> _labels,
                                 py::array_t<float> _K,
                                 py::array_t<float> _D,
                                 const float leaf_xy,
                                 const float leaf_z
                            ) {
  py::buffer_info buf_rgb = _rgb.request();
  py::buffer_info buf_depth = _depth.request();
  py::buffer_info buf_labels = _labels.request();

  assert(buf_labels.shape.size() == buf_depth.shape.size());
  for(int i = 0; i < buf_labels.shape.size(); ++i) 
    assert(buf_labels.shape.at(i)==buf_depth.shape.at(i));


  cv::Mat K, D; {
    float* K_ptr = (float*) _K.request().ptr;
    float* D_ptr = (float*) _D.request().ptr;
    K = (cv::Mat_<float>(3,3) << K_ptr[0], K_ptr[1], K_ptr[2],
                                 K_ptr[3], K_ptr[4], K_ptr[5],
                                 K_ptr[6], K_ptr[7], K_ptr[8]);

    D = cv::Mat::zeros(_D.size(),1, CV_32F);
    for(int i = 0; i < _D.size(); i++)
      D.at<float>(i,0) = D_ptr[i];
  }

  const int rows = buf_labels.shape[0];
  const int cols = buf_labels.shape[1];

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
  {
    const float min_depth = 0.0001;
    cv::Mat rgb(rows, cols, CV_8UC3);
    std::memcpy(rgb.data, buf_rgb.ptr, rows*cols*3*sizeof(unsigned char));
    std::vector<cv::Point2f> uv_points, normalized_points;
    std::vector<float> z_points;
    std::vector<cv::Scalar> colors;
    uv_points.reserve(rows*cols);
    colors.reserve(uv_points.capacity());
    z_points.reserve(uv_points.capacity());

    for (int r=0; r<rows; r++) {
      for (int c=0; c<cols; c++) {
        const float& d = ((float*) buf_depth.ptr)[r*cols+c];
        if( d < min_depth)
          continue;
        uv_points.push_back(cv::Point2f(c,r));
        z_points.push_back(d);
        const auto& pixel_rgb = rgb.at<cv::Vec3b>(r,c);
        cv::Scalar color(pixel_rgb[0],pixel_rgb[1],pixel_rgb[2]);
        colors.push_back(color);
      }
    }
    cv::undistortPoints(uv_points, normalized_points, K, D);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->reserve(uv_points.size());
    for(size_t i = 0; i < normalized_points.size(); i++){
      const cv::Point2f& xbar = normalized_points.at(i);
      const float& z = z_points.at(i);
      const cv::Scalar& color = colors.at(i);
      pcl::PointXYZRGB pt;
      pt.x = xbar.x*z;
      pt.y = xbar.y*z;
      pt.z = z;
      pt.r = (float)color[0];
      pt.g = (float)color[1];
      pt.b = (float)color[2];
      cloud->push_back(pt);
    }

    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf_xy, leaf_xy, leaf_z);
    sor.filter(*cloud_filtered);

    //std::cout << "filter " << cloud->size() << "->" << cloud_filtered->size() << std::endl;
  }

  const int n_points = cloud_filtered->size();
  py::array_t<float> xyzrgb(6*n_points);
  py::array_t<int32_t> ins_points(n_points);
  xyzrgb.resize({n_points, 6});
  ins_points.resize({n_points,});

  py::buffer_info buf_xyzrgb = xyzrgb.request();
  py::buffer_info buf_ins_points = ins_points.request();

  {
    std::vector<cv::Point3f> obj_points;
    obj_points.reserve(cloud_filtered->size());
    for(const pcl::PointXYZRGB& pt : *cloud_filtered)
      obj_points.push_back(cv::Point3f(pt.x,pt.y,pt.z));

    std::vector<cv::Point2f> img_points;
    cv::Mat tvec = cv::Mat::zeros(3,1,CV_32F);
    cv::Mat rvec = cv::Mat::zeros(3,1,CV_32F);
    cv::projectPoints(obj_points, rvec, tvec, K, D, img_points);
    int32_t* arr = (int32_t*) buf_ins_points.ptr;
    for(int i = 0; i < img_points.size(); i++){
      const cv::Point2i uv = img_points.at(i);
      if(uv.x<0 || uv.y <0 || uv.x >= cols || uv.y >= rows){
        arr[i] = 0;
      }
      else{
        const int r = uv.y;
        const int c = uv.x;
        arr[i] = ((int32_t*) buf_labels.ptr)[r*cols+c];
      }
    }
  }

  for(int i = 0; i < n_points; i++){
    const pcl::PointXYZRGB& pt = cloud_filtered->at(i);
    float* ptr = (float*)xyzrgb.request().ptr + 6*i;
    ptr[0] = pt.x;
    ptr[1] = pt.y;
    ptr[2] = pt.z;
    ptr[3] = ( (float) pt.r ) / 255.;
    ptr[4] = ( (float) pt.g ) / 255.;
    ptr[5] = ( (float) pt.b ) / 255.;
    for(int j = 3; j <6; j++){
      ptr[j] = std::min(1.f, ptr[j]);
      ptr[j] = std::max(0.f, ptr[j]);
    }
  }

  // ref: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html#instantiating-compound-python-types-from-c
  py::tuple output = py::make_tuple(xyzrgb, ins_points);
  return output;
}

PYBIND11_MODULE(unet_ext, m) {
  m.def("FindEdge", &PyFindEdge, "find edge", py::arg("input_mask") );
  m.def("GetGradient", &PyGetGradient, "get gradient", py::arg("input_depth"), py::arg("offset") );
  m.def("UnprojectPointscloud", &PyUnprojectPointscloud, "Get rgbxyz and xyzi points cloud",
        py::arg("rgb"), py::arg("depth"), py::arg("labels"), py::arg("K"), py::arg("D"), py::arg("leaf_xy"), py::arg("leaf_z"));
}

#else

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(unet_ext, m) {
  // https://developer.lsst.io/v/u-ktl-debug-fix/coding/python_wrappers_for_cpp_with_pybind11.html
  m.doc() = "Add two vectors using pybind11"; // optional module docstring
  m.def("add", &add, "A function which adds two numbers", pybind11::arg("i"), pybind11::arg("j")=20);

}

#endif
