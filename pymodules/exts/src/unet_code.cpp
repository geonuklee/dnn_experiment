#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <eigen3/Eigen/Dense> // TODO find_package at exts/py(2|3)ext/CMakeLists.txt

#include <opencv2/opencv.hpp>
#include <memory.h>
#include <vector>
#include <pcl/filters/voxel_grid.h>

namespace py = pybind11;
#ifndef WITHOUT_OBB
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/calib3d.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

std::vector<cv::Scalar> colors = {
  CV_RGB(0,255,0),
  CV_RGB(0,180,0),
  CV_RGB(0,100,0),
  CV_RGB(255,0,255),
  CV_RGB(100,0,255),
  CV_RGB(255,0,100),
  CV_RGB(100,0,100),
  CV_RGB(0,0,255),
  CV_RGB(0,0,180),
  CV_RGB(0,0,100),
  CV_RGB(255,255,0),
  CV_RGB(100,255,0),
  CV_RGB(255,100,0),
  CV_RGB(100,100,0),
  CV_RGB(255,0,0),
  CV_RGB(180,0,0),
  CV_RGB(100,0,0),
  CV_RGB(0,255,255),
  CV_RGB(0,100,255),
  CV_RGB(0,255,100),
  CV_RGB(0,100,100)
};

cv::Mat GetColoredLabel(cv::Mat marker){
  cv::Mat dst = cv::Mat::zeros(marker.rows, marker.cols, CV_8UC3);
  for(int r=0; r<marker.rows; r++){
    for(int c=0; c<marker.cols; c++){
      const int32_t& idx = marker.at<int32_t>(r,c);
      if(idx == 0)
        continue;
      const cv::Scalar& bgr = colors.at(idx % colors.size());
      dst.at<cv::Vec3b>(r,c)[0] = bgr[0];
      dst.at<cv::Vec3b>(r,c)[1] = bgr[1];
      dst.at<cv::Vec3b>(r,c)[2] = bgr[2];
    }
  }
  return dst;
}

template <typename K, typename T>
using EigenMap = std::map<K, T, std::less<K>, Eigen::aligned_allocator<std::pair<const K, T> > >;

struct OBB{
  float Tcb_xyz_qwxyz_[7];
  float scale_[3];
};

std::map<int, OBB> ComputeOBB(const std::vector<long int>& shape,
                              const int32_t* ptr_frontmarker,
                              const int32_t* ptr_marker,
                              const float* ptr_depth,
                              const EigenMap<int,Eigen::Matrix<float,4,1> >& label2vertices,
                              const float* ptr_numap,
                              const float* ptr_nvmap,
                              float max_depth
                             ){
  int rows = shape[0];
  int cols = shape[1];
  cv::Mat frontmarker(rows,cols, CV_32SC1, (void*)ptr_frontmarker);
  cv::Mat marker(rows, cols, CV_32SC1, (void*)ptr_marker);
  cv::Mat depth(rows, cols, CV_32F, (void*)ptr_depth);
  cv::Mat numap(rows, cols, CV_32F, (void*)ptr_numap);
  cv::Mat nvmap(rows, cols, CV_32F, (void*)ptr_nvmap);

  // Euclidean cluster before thickness measurement
  std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> allpoints;
  // For plane detection
  std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> frontpoints;

  auto uv2pclxyz = [&depth, &numap, &nvmap](int r, int c){
    float nu = numap.at<float>(r,c);
    float nv = nvmap.at<float>(r,c);
    const float& d = depth.at<float>(r,c);
    return pcl::PointXYZ(nu*d,nv*d,d);
  };
  auto uv2eigen_xyz = [&depth,&numap, &nvmap](float u, float v){
    int c = u;
    int r = v;
    float nu = numap.at<float>(r,c);
    float nv = nvmap.at<float>(r,c);
    const float& d = depth.at<float>(r,c);
    return Eigen::Vector3f(nu*d,nv*d,d);
  };


  // label 별, front points, side points,
  for(int r=0; r<rows; r++){
    for(int c=0; c<cols; c++){
      const int32_t& l_marker = marker.at<int32_t>(r,c);
      const int32_t& l_front = frontmarker.at<int32_t>(r,c);
      const float& d = depth.at<float>(r,c);
      if(d==0)
        continue;
      if(d > max_depth)
        continue;
      if(l_marker==0)
        continue;
      pcl::PointXYZ xyz = uv2pclxyz(r,c);
      if(l_front>0){
        if(!frontpoints.count(l_front))
          frontpoints[l_front] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>() );
        frontpoints.at(l_front)->push_back(xyz);
        if(!allpoints.count(l_front))
          allpoints[l_front] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>() );
        allpoints.at(l_front)->push_back(xyz);
      }
      else if(l_marker > 0){
        if(!allpoints.count(l_marker))
          allpoints[l_marker] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>() );
        allpoints.at(l_marker)->push_back(xyz);
      }
    }
  }
  // plane detection with frontpoints.
  const float leaf_size = 0.01;
  for(auto it : frontpoints){
    pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud = it.second;
    {
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(front_cloud);
      sor.setLeafSize(leaf_size,leaf_size,leaf_size);
      sor.filter(*front_cloud);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr all_cloud= allpoints.at(it.first);
    {
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(all_cloud);
      sor.setLeafSize(leaf_size,leaf_size,leaf_size);
      sor.filter(*all_cloud);
    }
  }

  const float inf = 999999.;
  std::map<int, OBB> output;
  // plane detection with frontpoints.
  for(auto it : frontpoints){
    pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud = it.second;
    // Do RANSACT for plane detection.
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr front_inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    //seg.setMethodType(pcl::SAC_LMEDS);
    seg.setDistanceThreshold(0.01);
    seg.setInputCloud(front_cloud);
    // The coefficients are consist with Hessian normal form : [normal_x normal_y normal_z d].
    // ref : https://pointclouds.org/documentation/group__sample__consensus.html
    seg.segment(*front_inliers, *coefficients);

    Eigen::Vector4f plane;
    for(int i =0; i <4; i++)
      plane[i] = coefficients->values.at(i);
    if(plane[2] > 0.5 ) // Normal vector of box supposed to be negative depth direction.
      plane = -plane;
    Eigen::Vector3f n = plane.head<3>();

    { // Filter front plane with inlier of RANSAC
      //std::cout << front_cloud->size() << "->";
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(front_cloud);
      extract.setIndices(front_inliers);
      extract.setNegative(false);
      extract.filter(*front_cloud);
      //std::cout << front_cloud->size() << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr all_cloud = allpoints.at(it.first);
    {
      const float euclidean_tolerance = 0.05;
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud(all_cloud);
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(euclidean_tolerance);
      ec.setMinClusterSize(0);
      ec.setMaxClusterSize(all_cloud->size());
      ec.setSearchMethod(tree);
      ec.setInputCloud(all_cloud);
      ec.extract(cluster_indices);
      std::sort(std::begin(cluster_indices),
                std::end(cluster_indices),
                [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
                return a.indices.size() > b.indices.size(); });

      for(int i = 0; i < cluster_indices.size(); i++){
        const pcl::PointIndices& inliers = cluster_indices[i];
        bool contact_with_plane = false;
        for(int j : inliers.indices){
          const auto& pt = all_cloud->at(j);
          float d = Eigen::Vector4f(pt.x,pt.y,pt.z,1.).dot(plane);
          if( std::abs(d) < 0.1){
            contact_with_plane = true;
            break;
          }
        }
        if(!contact_with_plane)
          continue;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(all_cloud);
        pcl::PointIndices::Ptr ptr(new pcl::PointIndices);
        ptr->indices = inliers.indices;
        extract.setIndices(ptr);
        extract.setNegative(false);
        extract.filter(*all_cloud);
        break;
      }
    }

    // Orientation from vertices
    // vertices : uv of o, y
    const Eigen::Matrix<float,4,1>& vertices = label2vertices.at(it.first);
    //Eigen::Vector3f o(0.,0., -plane[3]/plane[2]);
    Eigen::Vector3f pt_o = uv2eigen_xyz(vertices[0],vertices[1]);
    pt_o = pt_o - (pt_o.dot(n)+plane[3])*n;// projection on plane
    Eigen::Vector3f r1 = n;

    Eigen::Matrix<float,3,3> R0c;
    R0c.row(0) = r1.transpose();

    Eigen::Vector3f pt_a = uv2eigen_xyz(vertices[2],vertices[3]);
    pt_a = pt_a - (pt_a.dot(n)+plane[3])*n;// projection on plane
    Eigen::Vector3f delta = (pt_a - pt_o).normalized();
    Eigen::Vector3f r2, r3;
    if(std::abs(delta[1]) > std::abs(delta[0]) ){
      // pt_a should be y-axis
      r2 = delta[1] > 0. ? -delta : delta;
      r3 = r1.cross(r2);
    }
    else{
      // pt_a should be z-axis
      r3 = delta[0] > 0. ? -delta : delta;
      r2 = r3.cross(r1);
    }
    R0c.row(1) = r2.transpose();
    R0c.row(2) = r3.transpose();

    Eigen::Vector3f t0c; {
      Eigen::Matrix<float,3,3> Rc0 = R0c.transpose();
      Eigen::Vector3f tc0 = pt_o;
      t0c = - Rc0.transpose() * tc0;
    }
    // T0c -> Tbc,.. based on center position.
    g2o::SE3Quat T0c;
    T0c.setRotation(g2o::Quaternion(R0c.cast<double>() ) );
    T0c.setTranslation(t0c.cast<double>() );

    Eigen::Vector3d min_x0(inf, inf, inf);
    Eigen::Vector3d max_x0(0., -inf, -inf);
    // width height with front points
    for(const pcl::PointXYZ& pt : *front_cloud){
      Eigen::Vector3d x0 = T0c*Eigen::Vector3d(pt.x,pt.y,pt.z);
      for(int k=1; k<3; k++){
        min_x0[k] = std::min(min_x0[k], x0[k]);
        max_x0[k] = std::max(max_x0[k], x0[k]);
      }
    }
    // depth with all poitns.
    for(const pcl::PointXYZ& pt : *all_cloud){
      Eigen::Vector3d x0 = T0c*Eigen::Vector3d(pt.x,pt.y,pt.z);
      const int k = 0; // x-axis assigned for normal direction.
      min_x0[k] = std::min(min_x0[k], x0[k]);
    }

    if(all_cloud->size() < front_cloud->size()+10 ){
      const double min_depth = 0.2;
      if(min_x0[0] > -min_depth)
        min_x0[0] = -min_depth;
    }

    Eigen::Vector3f obb_size(max_x0[0]-min_x0[0], max_x0[1]-min_x0[1], max_x0[2]-min_x0[2]);
    g2o::SE3Quat T0b; // TODO
    T0b.setTranslation(Eigen::Vector3d(max_x0[0]+min_x0[0], max_x0[1]+min_x0[1], max_x0[2]+min_x0[2])/2);
#if 1
    g2o::SE3Quat Tcb = T0c.inverse() * T0b; // {c} <-> {box}
#else
    g2o::SE3Quat Tcb = T0c.inverse();
#endif

    // std::cout << Tcb.rotation().x() << std::endl;
    // std::cout << Tcb.to_homogeneous_matrix().block<3,4>(0,0) << ", and " << obb_size.transpose() << std::endl;
    OBB obb;
    for(int k = 0; k < 3; k++){
      obb.scale_[k] = obb_size[k];
      obb.Tcb_xyz_qwxyz_[k] = Tcb.translation()[k];
    }
    obb.Tcb_xyz_qwxyz_[3] = Tcb.rotation().w();
    obb.Tcb_xyz_qwxyz_[4] = Tcb.rotation().x();
    obb.Tcb_xyz_qwxyz_[5] = Tcb.rotation().y();
    obb.Tcb_xyz_qwxyz_[6] = Tcb.rotation().z();
    output[it.first] = obb;
  }
  return output;
}
#endif

bool SameSign(const float& v1, const float& v2){
  if(v1 > 0.)
    return v2 > 0.;
  else if(v1 < 0.)
    return v2 < 0.;
  return (v1 == 0.) && (v2 == 0.);
}

void GetDiscontinuousDepthEdge(const float* depth,
                               const std::vector<long int>& shape,
                               float threshold_depth,
                               unsigned char* output
                               ){
  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int size = rows*cols;
  const int hk = 1;
  memset((void*)output, false, size*sizeof(unsigned char) );

  for (int r0=hk; r0<rows-hk; r0++) {
    for (int c0=hk; c0<cols-hk; c0++) {
      const int idx0 = r0*cols+c0;
      const float& d_cp = depth[idx0];
      for(int r=r0-hk; r < r0+hk; r++){
        if(output[idx0])
          break;
        for(int c=c0-hk; c < c0+hk; c++){
          if(output[idx0])
            break;
          const float& d = depth[r*cols+c];
          bool c1 = d < 0.001;
          bool c2 = d_cp < 0.001;
          bool c3 = std::abs(d-d_cp) > threshold_depth;
          if(c1 || c2 || c3){
            output[idx0] = true;
            break;
          } // if abs(d-d_cp) > threshold
        } // for c
      } //for r
    } // for c0
  } // for r0
  return;
}


int GetFilteredDepth(const float* depth,
                     const unsigned char* dd_edge,
                     const std::vector<long int>& shape,
                     int sample_width,
                     float* filtered_depth) {
  //assert(sample_width > 2);
  assert(sample_width%2 == 1);

  const int rows = (int)shape.at(0);
  const int cols = (int)shape.at(1);
  const int size = rows*cols;
  const int hw = 2;
  memset((void*)filtered_depth, 0, 2*size*sizeof(float));

  enum SAMPLE_METHOD{
    MEAN,
    MEDIAN,
    MEAN_EXCLUDE_EXTREM
  };
  //SAMPLE_METHOD sample_method = sample_width > 4 ? MEAN_EXCLUDE_EXTREM : MEAN;
  SAMPLE_METHOD sample_method = MEAN;
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
    return 0.f;
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
      else if(sample_width < 3){
        zu = zv = d_cp;
      }
      else {
        su.clear();
        su.push_back(d_cp);
        // Sample depth for gx
        for(int dir = 0; dir < 2; dir++){
          for(int k=1; k<hk; k++){
            int r = dir==0? r0+k : r0-k;
            int idx = r*cols+c0;
            const float& d = depth[idx];
            const unsigned char& discontinuous = dd_edge[idx];
            if(discontinuous)
              break; // break for k
            su.push_back(d);
          }
        }
        zu = GetValue(su);

        // Sample depth for gy
        sv.clear();
        sv.push_back(d_cp);
        for(int dir = 0; dir < 2; dir++){
          for(int k=1; k<hk; k++){
            int c = dir==0? c0+k : c0-k;
            int idx = r0*cols+c;
            const float& d = depth[idx];
            const unsigned char& discontinuous = dd_edge[idx];
            if(discontinuous)
              break; // break for k
            sv.push_back(d);
          }
        }
        zv = GetValue(sv);
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
                 float sample_offset,
                 float fx,
                 float fy,
                 float* grad,
                 unsigned char* valid
                ){
  const int rows = shape[0];
  const int cols = shape[1];
  const int size = rows*cols;

  int boader = 10;
  //float doffset = 2*sample_offset;
  memset((void*)grad, 0, 2*size*sizeof(float));
  memset((void*)valid, 0, size*sizeof(unsigned char));

  for (int rc=sample_offset; rc<rows-sample_offset; rc++) {
    for (int cc=sample_offset; cc<cols-sample_offset; cc++) {
      // Remind : Gradients는 smoothing 때문에 gradient로 점진적으로 반응.
      int vidx0 = rc*cols+cc;
      int idx0 = 2*vidx0;

      const float& dcp = filtered_depth[idx0];
      const int pixel_offset = std::max<int>(1, (int) (fx*sample_offset/dcp) );
      int c0 = cc-pixel_offset;
      int r0 = rc-pixel_offset;
      int c1 = cc+pixel_offset;
      int r1 = rc+pixel_offset;
      if(c0 < 0 ||  r0 < 0 || c1 >= cols || r1 >= rows)
        continue;

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

      float du = c1 - c0;
      float dv = r1 - r0;
      float dx = du * dcp / fx;
      float dy = dv * dcp / fy;

      valid[vidx0] = true;
      grad[idx0] = (dx1 - dx0) / dx; // gx
      grad[idx0+1] = (dy1 - dy0) / dy; // gy
    }
  }
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
  {
  const int hk = 5;
  const float doffset = 2*hk;
  for (int rc=hk; rc<rows-hk; rc++) {
    for (int cc=hk; cc<cols-hk; cc++) {
      for(const auto& partial : {PARTIAL::X,PARTIAL::Y} ) {
        const int ic = rc*cols+cc;
        if(!valid[ic])
          continue;
          const int i0 = (partial==PARTIAL::X? rc*cols+cc-hk : (rc-hk)*cols+cc);
          const int i1 = (partial==PARTIAL::X? rc*cols+cc+hk : (rc+hk)*cols+cc);
          if(!valid[i0])
            continue;
          if(!valid[i1])
            continue;
          const float& g0 = (partial==PARTIAL::X? grad[2*i0] : grad[2*i0+1]);
          const float& g1 = (partial==PARTIAL::X? grad[2*i1] : grad[2*i1+1]);
          const float& dcp = depth[ic]; // Middle of sample point 0 and 1.
          float dxy = doffset * dcp / (partial==PARTIAL::X? fx : fy);
          float& h = (partial==PARTIAL::X? hessian_x[ic]:hessian_y[ic]);
          h = (g1-g0)/dxy;
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
                                 float sample_offset,
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
                                      py::array_t<unsigned char> dd_edge,
                                      int sample_width
                                     ) {
  py::buffer_info buf_inputdepth = inputdepth.request();
  py::buffer_info buf_ddedge = dd_edge.request();

  //  allocate the buffer, Depth map of $z_u$ and $z_v$
  py::array_t<float> output = py::array_t<float>(2*buf_inputdepth.size);
  py::buffer_info buf_output = output.request();
  float* ptr_output = (float*) buf_output.ptr;
  //memset((void*)ptr_output,0, 2*buf_inputdepth.size*sizeof(float));

  const float* ptr_inputdepth = (const float*) buf_inputdepth.ptr;
  const unsigned char* ptr_ddedge = (const unsigned char*) buf_ddedge.ptr;

  GetFilteredDepth(ptr_inputdepth,
                   ptr_ddedge,
                   buf_inputdepth.shape,
                   sample_width,
                   ptr_output);
  // reshape array to match input shape
  output.resize({buf_inputdepth.shape[0], buf_inputdepth.shape[1], 2L});
  return output;
}

py::array_t<float> PyGetDiscontinuousDepthEdge(py::array_t<float> inputdepth,
                                  float threshold_depth
                                  ) {
  py::buffer_info buf_inputdepth = inputdepth.request();
  py::array_t<unsigned char> output = py::array_t<unsigned char>(buf_inputdepth.size);
  py::buffer_info buf_output = output.request();
  unsigned char* ptr_output = (unsigned char*) buf_output.ptr;
  memset((void*)ptr_output,0, buf_inputdepth.size*sizeof(unsigned char));

  const float* ptr_inputdepth = (const float*) buf_inputdepth.ptr;
  GetDiscontinuousDepthEdge(ptr_inputdepth,
                            buf_inputdepth.shape,
                            threshold_depth,
                            ptr_output);

  output.resize({buf_inputdepth.shape[0], buf_inputdepth.shape[1]});
  return output;
}

#ifndef WITHOUT_OBB
py::list PyComputeOBB(py::array_t<int32_t> frontmarker,
                       py::array_t<int32_t> marker,
                       py::list py_label2vertices,
                       py::array_t<float> depth,
                       py::array_t<float> numap,
                       py::array_t<float> nvmap,
                       float max_depth
                       ){
  py::buffer_info buf_depth = depth.request();
  const float* ptr_depth = (const float*) buf_depth.ptr;

  py::buffer_info buf_marker = marker.request();
  const int32_t* ptr_marker = (const int32_t*) buf_marker.ptr;

  py::buffer_info buf_frontmarker = frontmarker.request();
  const int32_t* ptr_frontmarker = (const int32_t*) buf_frontmarker.ptr;

  py::buffer_info buf_numap = numap.request();
  const float* ptr_numap = (const float*) buf_numap.ptr;

  py::buffer_info buf_nvmap = nvmap.request();
  const float* ptr_nvmap = (const float*) buf_nvmap.ptr;

  EigenMap<int,Eigen::Matrix<float,4,1> > label2vertices; // org, x, y on 2D image plane

  size_t n = py_label2vertices.size();
  for(py::handle obj : py_label2vertices){
    assert(obj.attr("__class__").cast<py::str>().cast<std::string>() == "<type \'list\'>");
    py::list l = obj.cast<py::list>();

    int idx = l[0].cast<int>();
    py::array_t<float> vertices = l[1].cast<py::array_t<float> >();
    float* vertices_ptr = (float*) vertices.request().ptr;
    Eigen::Matrix<float,4,1> mat(vertices_ptr);
    label2vertices[idx] = mat;
  }
  //assert(info.contains("K"));
  //py::array_t<float> pyK = info["K"].cast<py::array_t<float> >();
  //cv::Mat K(3,3,CV_32FC1, pyK.request().ptr);
  std::map<int, OBB> output = ComputeOBB(buf_depth.shape,
                                         ptr_frontmarker,
                                         ptr_marker,
                                         ptr_depth,
                                         label2vertices,
                                         ptr_numap,
                                         ptr_nvmap,
                                         max_depth
                                         );
  py::list list;
  for(auto it : output){
    const OBB& obb = it.second;
    py::tuple pose = py::make_tuple(obb.Tcb_xyz_qwxyz_[0], obb.Tcb_xyz_qwxyz_[1],
                                    obb.Tcb_xyz_qwxyz_[2], obb.Tcb_xyz_qwxyz_[3],
                                    obb.Tcb_xyz_qwxyz_[4], obb.Tcb_xyz_qwxyz_[5],
                                    obb.Tcb_xyz_qwxyz_[6]);
    py::tuple scale = py::make_tuple(obb.scale_[0], obb.scale_[1], obb.scale_[2]);
    list.append(py::make_tuple(it.first, pose, scale) );
  }
  return list;
}
#endif


PYBIND11_MODULE(unet_ext, m) {
  m.def("GetFilteredDepth", &PyGetFilteredDepth, "Get filtered depth.",
        py::arg("input_mask"), py::arg("dd_edge"),
        py::arg("sample_width") );
  m.def("FindEdge", &PyFindEdge, "find edge", py::arg("input_mask") );
  m.def("GetDiscontinuousDepthEdge", &PyGetDiscontinuousDepthEdge,
        "Find edge of discontinuous depth",
        py::arg("input_depth"), py::arg("threshold_depth") );
  m.def("GetGradient", &PyGetGradient, "get gradient",
        py::arg("depth"), py::arg("sample_offset"),
        py::arg("fx"), py::arg("fy") );
  m.def("GetHessian", &PyGetHessian, "Get diagonal elements of Hessian",
        py::arg("depth"), py::arg("grad_sample_offset"), py::arg("grad_sample_width"),
        py::arg("fx"), py::arg("fy") );
  m.def("UnprojectPointscloud", &PyUnprojectPointscloud, "Get rgbxyz and xyzi points cloud",
        py::arg("rgb"), py::arg("depth"), py::arg("labels"), py::arg("K"), py::arg("D"), py::arg("leaf_xy"), py::arg("leaf_z"));

#ifndef WITHOUT_OBB
  m.def("ComputeOBB", &PyComputeOBB, "Compute OBB from given marker and depth map",
        py::arg("front_marker"),
        py::arg("marker"),
        py::arg("label2vertices"),
        py::arg("depth"),
        py::arg("nu_map"),
        py::arg("nv_map"),
        py::arg("max_depth")
        );
#endif
}
