#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
                 int sample_offset,
                 int sample_width,
                 float fx,
                 float fy,
                 float* grad,
                 bool* valid
                 ) {
  assert(sample_width > 2);
  assert(sample_width%2 == 1);
  const float max_depth_diff = 0.1; // TODO parameterize

  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int size = 2*rows*cols;
  for (int i=0; i < size; i++)
    grad[i] = 0.;
  if(valid)
    for (int i=0; i < size; i++)
      valid[i] = false;

  const int hk = (sample_width-1)/2;
  std::vector<float> samples0, samples;
  samples0.reserve(sample_width);
  samples.reserve(sample_width);

  enum SAMPLE_METHOD{
    MEAN,
    MEDIAN,
    MEAN_EXCLUDE_EXTREM
  };

  SAMPLE_METHOD sample_method = MEAN_EXCLUDE_EXTREM;

  auto GetValue = [&sample_method](std::vector<float>& values){
    if(sample_method == MEAN){
      // Mean - 실험결과 mean 이 더 정확했음.
      float sum = 0.;
      for(const float& v : values)
        sum += v;
      return sum / (float) values.size();
    }
    else if(sample_method == MEDIAN){
      std::sort(values.begin(), values.end());
      return values[values.size()/2];
    }
    else if(sample_method == MEAN_EXCLUDE_EXTREM){
      std::sort(values.begin(), values.end());
      const int extrem = values.size()/4;
      float sum = 0.;
      int k = 0;
      for(size_t i = extrem; i < values.size()-extrem; i++){
        k++;
        sum += values[k];
      }
      return sum / (float) k;
    }
    return -1.f;
  };

  int boader = sample_offset + hk + 1;
  for (int r0=boader; r0<rows-boader; r0++) {
    for (int c0=boader; c0<cols-boader; c0++) {
      const float& d_cp = depth[r0*cols+c0];
      int idx0 = 2*(r0*cols + c0);

      samples0.clear();   
      samples.clear();   
      { // Sample depth for gx
        int c = c0 + sample_offset;
        for(int r=r0-hk; r <= r0+hk; r++){
          const float& d0 = depth[r*cols+c0];
          if(std::abs(d_cp-d0) > max_depth_diff)
            break;
          else if(d0 > 0.)
            samples0.push_back(d0);
          else
            break;

          const float& sample = depth[r*cols+c];
          if(std::abs(d_cp-sample) > max_depth_diff)
            break;
          else if(sample > 0.)
            samples.push_back(sample);
          else
            break;
        }
      }

      if(samples0.size() == sample_width && samples.size() == sample_width) {
        float d0 = GetValue(samples0);
        float d1 = GetValue(samples);
        float dx = (float) sample_offset * d0 / fx;
        // gx
        grad[idx0] = (d1 - d0) / dx;
        if(valid)
          valid[idx0] = true;
      }

      samples0.clear();   
      samples.clear();   
      { // Compute depth for gy
        int r = r0 + sample_offset;
        for(int c=c0-hk; c <= c0+hk; c++){
          const float& d0 = depth[r0*cols+c];
          if(d0 > 0.)
            samples0.push_back(d0);
          else
            break;
          const float& sample = depth[r*cols+c];
          if(sample > 0.)
            samples.push_back(sample);
          else
            break;
        }
      }

      if(samples0.size() == sample_width && samples.size() == sample_width){
        float d0 = GetValue(samples0);
        float d1 = GetValue(samples);
        float dy = (float) sample_offset * d0 / fy;
        // gy
        grad[idx0+1] = (d1 - d0 ) / dy;
        if(valid)
          valid[idx0+1] = true;
      }
    }
  }
  return;
}

void GetHessian(const float* depth,
                  const std::vector<long int>& shape,
                  int grad_sample_offset,
                  int grad_sample_width,
                  float fx,
                  float fy,
                  float* hessian
                  ){
  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int size = rows*cols;

  float* grad = new float[2*size];
  bool* valid = new bool[2*size];
  GetGradient(depth, shape, grad_sample_offset, grad_sample_width, fx, fy, grad, valid);

  const float l0 = 0.;

  for (int i=0; i < size; i++)
    hessian[i] = l0;

  const int duv=1;
#if 1
  for (int r=duv; r < rows-duv; r++) {
    int rp = r+duv;
    int rn = r-duv;
    for (int c=duv; c < cols-duv; c++) {
      int cp = c+duv;
      int cn = c-duv;
      int idx = r*cols+c;
      const float& d0  = depth[idx];
      float gpx; {
        int j = 2*(r*cols+cp);
        if(!valid[j])
          continue;
        gpx = grad[j];
      }
      float gnx; {
        int j = 2*(r*cols+cn);
        if(!valid[j])
          continue;
        gnx = grad[j];
      }
      float gpy; {
        int j = 2*(rp*cols+c)+1;
        if(!valid[j])
          continue;
        gpy = grad[j];
      }
      float gny; {
        int j = 2*(rn*cols+c)+1;
        if(!valid[j])
          continue;
        gny = grad[j];
      }
      float dx = 2. * (float) duv * d0 / fx;
      float dy = 2. * (float) duv * d0 / fy;
      float lx = (gpx - gnx)/dx;
      float ly = (gpy - gny)/dy;
      hessian[idx] = lx + ly;
    }
  }
#else
  for (int r0=0; r0<rows; r0++) {
    for (int c0=0; c0<cols; c0++) {
      int idx = r0*cols + c0;
      int r = r0+duv;
      int c = c0+duv;
      const float& d0  = depth[r0*cols+c0];
      const float& d1x = depth[r0*cols+c ];
      const float& d1y = depth[ r*cols+c0];
      if(d0 == 0.)
        continue;

      float lx = l0;
      int xidx = 2*(r0*cols+c);
      bool bx = false;
      if(valid[2*idx] && valid[xidx]){
        float dx = (float) duv * d0 / fx;
        float gx = grad[xidx];
        const float& gx0 = grad[2*idx];
        lx = (gx - gx0) / dx;
        bx = true;
      }

      float ly = l0;
      int yidx = 2*(r*cols+c0) + 1;
      bool by = false;
      if(valid[2*idx+1] && valid[yidx]){
        float dy = (float) duv * d0 / fy;
        float gy = grad[yidx];
        const float& gy0 = grad[2*idx+1];
        ly = (gy - gy0) / dy;
        by = true;
      }

#if 0
      hessian[idx] = lx + ly;
#else
      if(bx && by)
        hessian[idx] = std::abs(lx) > std::abs(ly)? lx : ly;
      else if(bx)
        hessian[idx] = lx;
      else if(by)
        hessian[idx] = ly;
#endif
    }
  }
#endif

  delete[] grad;
  delete[] valid;
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

py::array_t<float> PyGetGradient(py::array_t<float> inputdepth,
                                 int sample_offset,
                                 int sample_width,
                                 float fx,
                                 float fy
                                 ) {
  py::buffer_info buf_inputdepth = inputdepth.request();

  /*  allocate the buffer */
  py::array_t<float> output = py::array_t<float>(2*buf_inputdepth.size);
  py::buffer_info buf_output = output.request();

  const float* ptr_inputdepth = (const float*) buf_inputdepth.ptr;
  float* ptr_output = (float*) buf_output.ptr;
  GetGradient(ptr_inputdepth, buf_inputdepth.shape, sample_offset, sample_width, fx, fy, ptr_output, nullptr);

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

py::array_t<float> PyGetHessian(py::array_t<float> inputdepth,
                                  int grad_sample_offset,
                                  int grad_sample_width,
                                  float fx,
                                  float fy
                                  ) {
  py::buffer_info buf_inputdepth = inputdepth.request();
  /*  allocate the buffer */
  py::array_t<float> output = py::array_t<float>(buf_inputdepth.size);
  py::buffer_info buf_output = output.request();

  const float* ptr_inputdepth = (const float*) buf_inputdepth.ptr;
  float* ptr_output = (float*) buf_output.ptr;
  GetHessian(ptr_inputdepth,
               buf_inputdepth.shape,
               grad_sample_offset,
               grad_sample_width,
               fx,
               fy,
               ptr_output);

  // reshape array to match input shape
  output.resize({buf_inputdepth.shape[0], buf_inputdepth.shape[1]});
  return output;
}

PYBIND11_MODULE(unet_ext, m) {
  m.def("FindEdge", &PyFindEdge, "find edge", py::arg("input_mask") );
  m.def("GetGradient", &PyGetGradient, "get gradient",
        py::arg("depth"), py::arg("sample_offset"), py::arg("sample_width"),
        py::arg("fx"), py::arg("fy") );
  m.def("GetHessian", &PyGetHessian, "Get diagonal elements of Hessian",
        py::arg("depth"), py::arg("grad_sample_offset"), py::arg("grad_sample_width"),
        py::arg("fx"), py::arg("fy") );
  m.def("UnprojectPointscloud", &PyUnprojectPointscloud, "Get rgbxyz and xyzi points cloud",
        py::arg("rgb"), py::arg("depth"), py::arg("labels"), py::arg("K"), py::arg("D"), py::arg("leaf_xy"), py::arg("leaf_z"));
}
