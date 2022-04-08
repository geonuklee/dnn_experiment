#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <eigen3/Eigen/Dense> // TODO find_package at exts/py(2|3)ext/CMakeLists.txt

#include <opencv2/opencv.hpp>
#include <memory.h>
#include <vector>
#include <pcl/filters/voxel_grid.h>

namespace py = pybind11;

bool SameSign(const float& v1, const float& v2){
  if(v1 > 0.)
    return v2 > 0.;
  else if(v1 < 0.)
    return v2 < 0.;
  return (v1 == 0.) && (v2 == 0.);
}

int GetFilteredDepth(const float* depth,
                     const std::vector<long int>& shape,
                     int sample_width,
                     float* filtered_depth) {
  //assert(sample_width > 2);
  assert(sample_width%2 == 1);

  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int hw = 2;
  const float max_depth_diff = 0.1; // TODO parameterize

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

  const int hk = (sample_width-1)/2;

  std::vector<float> su, sv;
  su.reserve(sample_width);
  sv.reserve(sample_width);

  for (int r0=hk; r0<rows-hk; r0++) {
    for (int c0=hk; c0<cols-hk; c0++) {
      const float& d_cp = depth[r0*cols+c0];
      float zu, zv;
      if(d_cp ==0.){
        zu = zv = 0.;
      }
      else if(sample_width > 2){
        su.clear();
        // Sample depth for gx
        for(int r=r0-hk; r <= r0+hk; r++){
          const float& d = depth[r*cols+c0];
          if(std::abs(d_cp-d) > max_depth_diff)
            continue;
          else if(d > 0.)
            su.push_back(d);
          else
            continue;
        }
        zu = GetValue(su);
        // Sample depth for gy
        sv.clear();
        for(int c=c0-hk; c <= c0+hk; c++){
          const float& d = depth[r0*cols+c];
          if(std::abs(d_cp-d) > max_depth_diff)
            continue;
          else if(d > 0.)
            sv.push_back(d);
          else
            continue;
        }
        zv = GetValue(sv);
      }
      else{
        zu = zv = d_cp;
      }

      int idx0 = 2*(r0*cols + c0);
      filtered_depth[idx0] = zu;
      filtered_depth[idx0+1] = zv;
    }
  }
  //printf("GetFilteredDepth %d, %d \n", rows, cols);
  return 0;
}

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
      arr_out[r0*cols + c0] = (is_edge?1:0);
    }
  }
  return 0;
}

void GetGradient(const float* filtered_depth,
                 const std::vector<long int>& shape,
                 int sample_offset,
                 float fx,
                 float fy,
                 float* grad,
                 unsigned char* valid
                ){
  int rows = shape[0];
  int cols = shape[1];

  int boader = sample_offset;
  float doffset = 2*sample_offset;
  for (int rc=sample_offset; rc<rows-sample_offset; rc++) {
    for (int cc=sample_offset; cc<cols-sample_offset; cc++) {
      // Remind : Gradients는 smoothing 때문에 gradient로 점진적으로 반응.
      int vidx0 = rc*cols+cc;
      int idx0 = 2*vidx0;
      grad[idx0] = grad[idx0+1] = 0.f;

      const float& dcp = filtered_depth[idx0];
      int c0 = cc-sample_offset;
      int r0 = rc-sample_offset;
      int c1 = cc+sample_offset;
      int r1 = rc+sample_offset;

      const float& dx0 = filtered_depth[2*(rc*cols+c0)];
      const float& dx1 = filtered_depth[2*(rc*cols+c1)];
      const float& dy0 = filtered_depth[2*(r0*cols+cc)+1];
      const float& dy1 = filtered_depth[2*(r1*cols+cc)+1];

      if(dcp == 0.f)
        continue;
      else if(dx0 == 0.f)
        continue;
      else if(dx1 == 0.f)
        continue;
      else if(dy0 == 0.f)
        continue;
      else if(dy1 == 0.f)
        continue;

      float dx = doffset * dcp / fx;
      float dy = doffset * dcp / fy;

      valid[vidx0] = true;
      grad[idx0] = (dx1 - dx0) / dx; // gx
      grad[idx0+1] = (dy1 - dy0) / dy; // gy
    }
  }
  return;
}

void GetHessianWithFixedSamplePosition(const float* depth,
                const float* grad,
                const unsigned char* valid,
                const std::vector<long int>& shape,
                float fx,
                float fy,
                float* hessian)
{
  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int size = rows*cols;
  // TODO -논문용- 왜 두겹짜리 Hessian edge가 발생하는지?
  // Done 해결책 구현
  int hk = 20;
  const float vmax = 999999.;

  float* hessian_x = new float[size];
  float* hessian_y = new float[size];
  memset((void*)hessian_x, 0, size*sizeof(float) );
  memset((void*)hessian_y, 0, size*sizeof(float) );

  enum PARTIAL {X=0, Y=1};
  for (int rc=hk; rc<rows-hk; rc++) {
    for (int cc=hk; cc<cols-hk; cc++) {
      // Partial differential with fixed sampling point.
      const float& gx0 = grad[2*(rc*cols+cc-1)];
      const float& gx1 = grad[2*(rc*cols+cc+1)];
      const float du = 2;
      const int i = rc*cols+cc;
      const float& dcp = depth[i];
      const float dx = du * dcp / fx;
      const float hx = (gx1 - gx0) / dx;
      hessian_x[i] = hx;
    }
  }
  for(int i = 0; i < size; i++)
    hessian[i] = hessian_x[i] + hessian_y[i];

  delete[] hessian_x;
  delete[] hessian_y;
  return;
}

void GetHessian(const float* depth,
                const float* grad,
                const unsigned char* valid,
                const std::vector<long int>& shape,
                float fx,
                float fy,
                float* hessian)
{
  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int size = rows*cols;
  const float vmax = 999999.;

  float* hessian_x = new float[size];
  float* hessian_y = new float[size];
  memset((void*)hessian_x, 0, size*sizeof(float) );
  memset((void*)hessian_y, 0, size*sizeof(float) );

  enum PARTIAL {X, Y};
  bool* bmax_x = new bool[size];
  bool* bmax_y = new bool[size];
  memset((void*)bmax_x, true, size*sizeof(bool) );
  memset((void*)bmax_y, true, size*sizeof(bool) );

  // 실험결과 multi scale offset은 random에서, min offset은 공통 필요했음.
  const int hd = 6;
  for (int rc=hd; rc<rows-hd; rc++) {
    for (int cc=hd; cc<cols-hd; cc++) {
      for(const auto& partial : {PARTIAL::X,PARTIAL::Y} ) {
        const int ic = rc*cols+cc;
        if(!valid[ic])
          continue;
        for(int k = hd-1; k < hd; k++){
          const int i0 = (partial==PARTIAL::X? rc*cols+cc-k : (rc-k)*cols+cc);
          const int i1 = (partial==PARTIAL::X? rc*cols+cc+k : (rc+k)*cols+cc);
          if(!valid[i0])
            continue;
          if(!valid[i1])
            continue;
          const float& g0 = (partial==PARTIAL::X? grad[2*i0] : grad[2*i0+1]);
          const float& g1 = (partial==PARTIAL::X? grad[2*i1] : grad[2*i1+1]);
          const float& dcp = depth[ic]; // Middle of sample point 0 and 1.
          float doffset = 2 * k;
          float dxy = doffset * dcp / (partial==PARTIAL::X? fx : fy);
          float& h = (partial==PARTIAL::X? hessian_x[ic]:hessian_y[ic]);
          float h_candidate = (g1-g0)/dxy;
          if(std::abs(h_candidate) > std::abs(h))
            h = h_candidate;
        }
      }
    }
  }

  for(int i = 0; i < size; i++)
    hessian[i] = hessian_x[i] + hessian_y[i];

  const int hk = 10;
  for (int rc=hk; rc<rows-hk; rc++) {
    for (int cc=hk; cc<cols-hk; cc++) {
      const int ic = rc*cols+cc;
      if(!valid[ic])
        continue;
      const float& hc = hessian[ic];
      const float abs_hc = std::abs(hc);

      bool& is_maximum = bmax_x[ic];
      bool search_dir[4] = {true,true,true,true};
      for(int k=1; k< hk; k++){
        for(const auto& partial : {PARTIAL::X,PARTIAL::Y} ) {
          const int i0 = (partial==PARTIAL::X? rc*cols+cc-k : (rc-k)*cols+cc);
          const int i1 = (partial==PARTIAL::X? rc*cols+cc+k : (rc+k)*cols+cc);
          if(search_dir[partial] ){
            if(valid[i0]) {
              const float& h0 = hessian[i0];
              if( (std::abs(h0) > abs_hc) && (!SameSign(h0,hc)) ){
                is_maximum = false;
                break; // break for loop of k
              }
            } else {
              // TODO invalid 또는 depth 오차가 급격히 증가했을때 - 즉 실루엣
              search_dir[partial] = false;
            }
          }
          if(search_dir[2+partial] ){
            if(valid[i1]) {
              const float& h1 = hessian[i1];
              if( (std::abs(h1) > abs_hc) && (!SameSign(h1,hc)) ){
                is_maximum = false;
                break; // break for loop of k
              }
            } else {
              search_dir[2+partial] = false;
            }
          }
        }
        if(!is_maximum)
          break;
      }
    }
  }

  for (int r=1; r<rows-1; r++) {
    for (int c=1; c<cols-1; c++) {
      const int i = r*cols+c;
      if(!bmax_x[i])
        hessian[i] = 0.;
    }
  }
  delete[] bmax_x;
  delete[] bmax_y;

  delete[] hessian_x;
  delete[] hessian_y;
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

py::tuple PyGetGradient(py::array_t<float> filtered_depth,
                                 int sample_offset,
                                 float fx,
                                 float fy
                                 ) {
  py::buffer_info buf_filtered_depth = filtered_depth.request();
  long rows = buf_filtered_depth.shape[0];
  long cols = buf_filtered_depth.shape[1];

  /*  allocate the buffer */
  py::array_t<float> grad = py::array_t<float>(rows*cols*2);
  py::array_t<unsigned char> valid = py::array_t<unsigned char>(rows*cols);
  py::buffer_info buf_grad = grad.request();
  float* ptr_grad = (float*) buf_grad.ptr;

  py::buffer_info buf_valid = valid.request();
  unsigned char* ptr_valid = (unsigned char*) buf_valid.ptr;
  memset((void*)ptr_valid,0, buf_valid.size*sizeof(unsigned char));

  const float* ptr_filtered_depth = (const float*) buf_filtered_depth.ptr;
  GetGradient(ptr_filtered_depth, buf_filtered_depth.shape,
              sample_offset, fx, fy, ptr_grad, ptr_valid);

  // reshape array to match input shape
  grad.resize({rows, cols, 2L});
  valid.resize({rows, cols});

  // ref: https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html#instantiating-compound-python-types-from-c
  py::tuple output = py::make_tuple(grad, valid);
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

py::array_t<float> PyGetHessian(py::array_t<float> depth,
                                py::array_t<float> grad,
                                py::array_t<unsigned char> valid,
                                float fx,
                                float fy
                               ) {
  py::buffer_info buf_depth = depth.request();
  long rows = buf_depth.shape[0];
  long cols = buf_depth.shape[1];
  const float* ptr_depth = (const float*) buf_depth.ptr;

  py::buffer_info buf_grad = grad.request();
  const float* ptr_grad = (const float*) buf_grad.ptr;

  py::buffer_info buf_valid = valid.request();
  const unsigned char* ptr_valid = (const unsigned char*) buf_valid.ptr;

  /*  allocate the buffer */
  py::array_t<float> hessian = py::array_t<float>(rows*cols);
  py::buffer_info buf_hessian = hessian.request();
  float* ptr_hessian = (float*) buf_hessian.ptr;
  memset((void*)ptr_hessian,0, buf_hessian.size*sizeof(float));

  GetHessian(ptr_depth,
             ptr_grad,
             ptr_valid,
             buf_depth.shape,
             fx,
             fy,
             ptr_hessian);

  // reshape array to match input shape
  hessian.resize({rows, cols});
  return hessian;
}

py::array_t<float> PyGetFilteredDepth(py::array_t<float> inputdepth,
                                  int sample_width
                                  ) {
  py::buffer_info buf_inputdepth = inputdepth.request();
  //  allocate the buffer, Depth map of $z_u$ and $z_v$
  py::array_t<float> output = py::array_t<float>(2*buf_inputdepth.size);
  py::buffer_info buf_output = output.request();
  float* ptr_output = (float*) buf_output.ptr;
  //memset((void*)ptr_output,0, 2*buf_inputdepth.size*sizeof(float));

  const float* ptr_inputdepth = (const float*) buf_inputdepth.ptr;

  GetFilteredDepth(ptr_inputdepth,
                   buf_inputdepth.shape,
                   sample_width,
                   ptr_output);
  // reshape array to match input shape
  output.resize({buf_inputdepth.shape[0], buf_inputdepth.shape[1], 2L});
  return output;
}


PYBIND11_MODULE(unet_ext, m) {
  m.def("GetFilteredDepth", &PyGetFilteredDepth, "Get filtered depth.", py::arg("input_mask"),
        py::arg("sample_width") );
  m.def("FindEdge", &PyFindEdge, "find edge", py::arg("input_mask") );
  m.def("GetGradient", &PyGetGradient, "get gradient",
        py::arg("depth"), py::arg("sample_offset"),
        py::arg("fx"), py::arg("fy") );
  m.def("GetHessian", &PyGetHessian, "Get diagonal elements of Hessian",
        py::arg("depth"), py::arg("grad_sample_offset"), py::arg("grad_sample_width"),
        py::arg("fx"), py::arg("fy") );
  m.def("UnprojectPointscloud", &PyUnprojectPointscloud, "Get rgbxyz and xyzi points cloud",
        py::arg("rgb"), py::arg("depth"), py::arg("labels"), py::arg("K"), py::arg("D"), py::arg("leaf_xy"), py::arg("leaf_z"));
}
