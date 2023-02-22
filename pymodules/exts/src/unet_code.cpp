#include <iostream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <eigen3/Eigen/Dense> // TODO find_package at exts/py(2|3)ext/CMakeLists.txt
#include <eigen3/Eigen/Geometry>

#include <opencv2/opencv.hpp>
#include <memory.h>
#include <vector>
#include <pcl/filters/voxel_grid.h>

namespace py = pybind11;
#include <g2o/types/slam3d/se3quat.h>
#include <opencv2/calib3d.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/search.hpp> // https://github.com/PointCloudLibrary/pcl/issues/2406
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

template<typename T>
cv::Mat PyArray2Cv(const py::array_t<T>& pyarray, int cv_type){
  py::buffer_info buf = pyarray.request();
  long rows = buf.shape[0];
  long cols = buf.shape[1];
  cv::Mat output(rows,cols,cv_type,(void*)buf.ptr);
  return output;
}

template<typename T, int R, int C>
Eigen::Matrix<T,R,C> PyArray2Eigen(const py::array_t<T>& pyarray){
  py::buffer_info buf = pyarray.request();
  Eigen::Matrix<T,R,C, Eigen::RowMajor> mat((T*)buf.ptr);
  return mat;
}

template<typename T>
py::array_t<T> Cv2PyArray(cv::Mat mat){
  py::array_t<T> pyarray(mat.rows*mat.cols*mat.channels());
  py::buffer_info buf = pyarray.request();
  T* ptr = (T*) buf.ptr;
  std::memcpy(ptr, mat.data, mat.rows*mat.cols*mat.channels()*sizeof(T));
  if(mat.channels()==1)
    pyarray.resize({mat.rows, mat.cols});
  else
    pyarray.resize({mat.rows, mat.cols, mat.channels() });
  return pyarray;
}

cv::Mat GetBoundary(const cv::Mat marker, int w);
cv::Mat GetColoredLabel(cv::Mat mask, bool put_text=false){
  cv::Mat dst = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);
  std::map<int, cv::Point> annotated_lists;
  //std::map<int, int> max_area;

  cv::Mat connected_labels, stats, centroids;
  cv::Mat binary = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
  if(mask.type() == CV_8UC1){
    for(size_t i = 0; i < mask.rows; i++){
      for(size_t j = 0; j < mask.cols; j++){
        int idx = mask.at<unsigned char>(i,j);
        if(idx > 1)
          binary.at<unsigned char>(i,j) = 1;
      }
    }
  }
  else if(mask.type() == CV_32SC1){
    cv::Mat boundary = GetBoundary(mask,2);
    for(int i = 0; i < mask.rows; i++)
      for(int j = 0; j < mask.cols; j++)
        binary.at<unsigned char>(i,j) = boundary.at<unsigned char>(i,j) <1;
  }
  else
    throw "Unexpected type";
  cv::connectedComponentsWithStats(binary, connected_labels, stats, centroids);

  for(int i=1; i<stats.rows; i++) {
    int x = centroids.at<double>(i, 0);
    int y = centroids.at<double>(i, 1);
    if(x < 0 or y < 0 or x >= mask.cols or y >= mask.cols)
      continue;
    const int& x0 = stats.at<int>(i,cv::CC_STAT_LEFT);
    const int& x1 = x0+stats.at<int>(i,cv::CC_STAT_WIDTH);
    const int& y0 = stats.at<int>(i,cv::CC_STAT_TOP);
    const int& y1 = y0+stats.at<int>(i,cv::CC_STAT_HEIGHT);
    const int& area = stats.at<int>(i,cv::CC_STAT_AREA);
    int idx;
    if(mask.type() == CV_8UC1)
      idx = mask.at<unsigned char>(y,x);
    else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
      idx = mask.at<int>(y,x);
    //if(idx > 1 && area > max_area[idx] )
    if(idx > 1){
      cv::Point pt( (x0+x1)/2, (y0+y1)/2);
      annotated_lists[idx] = pt;
      //max_area[idx] = area;
    }
  }

  for(size_t i = 0; i < mask.rows; i++){
    for(size_t j = 0; j < mask.cols; j++){
      int idx;
      if(mask.type() == CV_8UC1)
        idx = mask.at<unsigned char>(i,j);
      else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
        idx = mask.at<int>(i,j);
      else
        throw "Unexpected type";
      if(mask.type() == CV_8UC1 && idx == 0)
        continue;
      else if(mask.type() == CV_32S && idx < 0)
        continue;

      cv::Scalar bgr;
      if( idx == 0)
        bgr = CV_RGB(100,100,100);
      else if (idx == 1)
        bgr = CV_RGB(255,255,255);
      else
        bgr = colors.at( idx % colors.size() );

      dst.at<cv::Vec3b>(i,j)[0] = bgr[0];
      dst.at<cv::Vec3b>(i,j)[1] = bgr[1];
      dst.at<cv::Vec3b>(i,j)[2] = bgr[2];

      if(idx > 1 && !annotated_lists.count(idx) ){
        bool overlaped=false;
        cv::Point pt(j,i+10);
        for(auto it : annotated_lists){
          cv::Point e(pt - it.second);
          if(std::abs(e.x)+std::abs(e.y) < 20){
            overlaped = true;
            break;
          }
        }
        if(!overlaped)
          annotated_lists[idx] = pt;
      }
    }
  }

  if(put_text){
    for(auto it : annotated_lists){
      //cv::rectangle(dst, it.second+cv::Point(0,-10), it.second+cv::Point(20,0), CV_RGB(255,255,255), -1);
      const auto& c0 = colors.at( it.first % colors.size() );
      //const auto color = c0;
      //const auto color = (c0[0]+c0[1]+c0[2] > 255*2) ? CV_RGB(0,0,0) : CV_RGB(255,255,255);
      const auto color = CV_RGB(255-c0[2],255-c0[1],255-c0[0]);
      cv::putText(dst, std::to_string(it.first), it.second, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
  }
  return dst;
}



template<typename PointT>
void EuclideanCluster(const float euclidean_tolerance, boost::shared_ptr<pcl::PointCloud<PointT> > all_cloud){
  boost::shared_ptr< pcl::search::KdTree<PointT> > tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(all_cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(euclidean_tolerance);
  ec.setMinClusterSize(0);
  ec.setMaxClusterSize(all_cloud->size());
  ec.setSearchMethod(tree);
  ec.setInputCloud(all_cloud);
  ec.extract(cluster_indices); // << Failure with pybind11 at python3

  std::sort(std::begin(cluster_indices),
            std::end(cluster_indices),
            [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() > b.indices.size(); });

  for(int i = 0; i < cluster_indices.size(); i++){
    const pcl::PointIndices& inliers = cluster_indices[i];
    //bool contact_with_plane = false;
    //for(int j : inliers.indices){
    //  const auto& pt = all_cloud->at(j);
    //  float d = Eigen::Vector4f(pt.x,pt.y,pt.z,1.).dot(plane);
    //  if( std::abs(d) < 0.1){
    //    contact_with_plane = true;
    //    break;
    //  }
    //}
    //if(!contact_with_plane)
    //  continue;
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(all_cloud);
    pcl::PointIndices::Ptr ptr(new pcl::PointIndices);
    ptr->indices = inliers.indices;
    extract.setIndices(ptr);
    extract.setNegative(false);
    extract.filter(*all_cloud);
    break;
  }
  return;
}

template <typename K, typename T>
using EigenMap = std::map<K, T, std::less<K>, Eigen::aligned_allocator<std::pair<const K, T> > >;

template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T> >;

struct OBB{
  float Tcb_xyz_qwxyz_[7];
  float scale_[3];
};

EigenVector<Eigen::Vector4f> GetFrontSidePlanes(const OBB& obb){
  const float* qt = obb.Tcb_xyz_qwxyz_;
  Eigen::Quaterniond quat(qt[3],qt[4],qt[5],qt[6]);
  Eigen::Vector3d t(qt[0],qt[1],qt[2]);
  g2o::SE3Quat Tcb(quat,t);
  g2o::SE3Quat Tbc = Tcb.inverse();
  Eigen::Matrix3d Rbc = Tbc.rotation().matrix();

  // TODO Hard coded coordinate transform, bx~=cz, by~=-cy, bz~=-cx
  std::vector<int> axis_c2b = {2,1,0};

  EigenVector<Eigen::Vector4f> output;
  output.reserve(5); {
      Eigen::Vector4f plane;
      const int& k = axis_c2b.at(2);
      Eigen::Vector3d n = Rbc.row(k);
      plane.head<3>() = Eigen::Vector3f(n[0],n[1],n[2]);
      Eigen::Vector3d pt0;
      pt0[k] = obb.scale_[k] / 2.;
      Eigen::Vector3d pt = Tcb * pt0;
      plane[3] = -n.dot(pt);
      output.push_back(plane);
      //std::cout << "plane #0 = " << plane.transpose() << std::endl;
  }

  for(int axis = 0; axis < 2; axis++){
    for(int i = 0; i < 2; i++){
      const int& k = axis_c2b.at(axis);
      float sign = i==0 ? 1. : -1.;
      Eigen::Vector4f plane;
      Eigen::Vector3d n = sign * Rbc.row(k);
      plane.head<3>() = Eigen::Vector3f(n[0],n[1],n[2]);
      Eigen::Vector3d pt0;
      pt0[k] = sign * obb.scale_[k] / 2.;
      Eigen::Vector3d pt = Tcb * pt0;
      plane[3] = -n.dot(pt);
      output.push_back(plane);
      //std::cout << "plane #" << output.size()-1 << " = " << plane.transpose() << std::endl;
    }
  }

  return output;
}

std::map<int, OBB> ComputeOBB(cv::Mat frontmarker,
                              cv::Mat marker,
                              cv::Mat depth,
                              const EigenMap<int,Eigen::Matrix<float,4,1> >& label2vertices,
                              cv::Mat numap,
                              cv::Mat nvmap,
                              float max_depth
                             ){
  int rows = marker.rows;
  int cols = marker.cols;

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
  const float leaf_size = 0.005;
  const float euclidean_tolerance = 0.01;
  for(auto it : frontpoints){
    pcl::PointCloud<pcl::PointXYZ>::Ptr front_cloud = it.second;
    pcl::PointCloud<pcl::PointXYZ>::Ptr all_cloud= allpoints.at(it.first);
    {
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(front_cloud);
      sor.setLeafSize(leaf_size,leaf_size,leaf_size);
      sor.filter(*front_cloud);
    }
    {
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(all_cloud);
      sor.setLeafSize(leaf_size,leaf_size,leaf_size);
      sor.filter(*all_cloud);
    }
    EuclideanCluster<pcl::PointXYZ>(euclidean_tolerance, front_cloud);
    EuclideanCluster<pcl::PointXYZ>(euclidean_tolerance, all_cloud);
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
    seg.setDistanceThreshold(0.005);
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

    if(max_x0[2]-min_x0[2] < .01 ){
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

std::set<std::pair<int,int> > GetNeighbors(cv::Mat marker, int w){
  const int bg = 0;
  std::set<std::pair<int,int> > contacts;
  for (int r0=0; r0<marker.rows; r0++) {
    for (int c0=0; c0<marker.cols; c0++) {
      const int32_t& i0 = marker.at<int>(r0,c0);
      if(i0 == bg)
        continue;
      for(int r1 = std::max(r0-w,0); r1 < std::min(r0+w,marker.rows); r1++){
        for(int c1 = std::max(c0-w,0); c1 < std::min(c0+w,marker.cols); c1++){
          const int32_t& i1 = marker.at<int>(r1,c1);
          if(i1 == bg)
            continue;
          if(i0 == i1)
            continue;
          std::pair<int,int> contact(std::min(i0,i1), std::max(i0,i1) );
          contacts.insert(contact);
        }
      }
    }
  }
  //for(const auto& contact : contacts)
  //  std::cout << contact.first << ", " << contact.second << std::endl;
  return contacts;
}

std::set<std::pair<int,int> > GetNeighbors(cv::Mat plane_marker, int w,
                                           const std::map<int,int>& plane2marker,
                                           bool limit_nearest_plane,
                                           cv::Mat& opposite){
  const int bg = 0;
  std::set<std::pair<int,int> > contacts;
  opposite = cv::Mat::zeros(plane_marker.rows, plane_marker.cols, CV_32SC1);
  cv::Mat dist = cv::Mat::zeros(plane_marker.rows, plane_marker.cols, CV_32F);
  const float r2 = (w+1)*(w+1);
  for (int r0=0; r0<plane_marker.rows; r0++) {
    for (int c0=0; c0<plane_marker.cols; c0++) {
      const int& i0 = plane_marker.at<int>(r0,c0);
      int& o0 = opposite.at<int>(r0,c0);
      if(i0 == bg)
        continue;
      if(!plane2marker.count(i0))
        continue;

      const int& m0 = plane2marker.at(i0);
      for(int r1 = std::max(r0-w,0); r1 < std::min(r0+w,plane_marker.rows); r1++){
        for(int c1 = std::max(c0-w,0); c1 < std::min(c0+w,plane_marker.cols); c1++){
          const int& i1 = plane_marker.at<int>(r1,c1);
          int& o1 = opposite.at<int>(r1,c1);
          if(i1 == bg)
            continue;
          if(!plane2marker.count(i1))
            continue;

          if(i0 == i1)
            continue;
          const int& m1 = plane2marker.at(i1);
          if(m0 == m1 && !limit_nearest_plane)
            continue;
          float dx = c1-c0;
          float dy = r1-r0;
          float d = dx*dx+dy*dy;
          if(d > r2)
            continue;
          std::pair<int,int> contact(std::min(i0,i1), std::max(i0,i1) );
          contacts.insert(contact);
          if(o0 == 0){
            dist.at<float>(r0,c0) = d;
            o0 = i1;
          }
          else if(d < dist.at<float>(r0,c0) ){
            dist.at<float>(r0,c0) = d;
            o0 = i1;
          }
          if(o1 == 0){
            dist.at<float>(r1,c1) = d;
            o1 = i0;
          }
          else if(d < dist.at<float>(r1,c1) ){
            dist.at<float>(r1,c1) = d;
            o1 = i0;
          }
        }
      }
    }
  }

  if(limit_nearest_plane){
    for(int r=0; r<plane_marker.rows;r++){
      for(int c=0; c<plane_marker.cols;c++){
        int& pidx1 = opposite.at<int>(r,c);
        if(pidx1 == 0)
          continue;
        const int& pidx0 = plane_marker.at<int>(r,c);
        const int& midx0 = plane2marker.at(pidx0);
        const int& midx1 = plane2marker.at(pidx1);
        if(midx0==midx1) // Erase boundary between same instance with same marker.
          pidx1 = 0;
      }
    }
  }
  return contacts;
}

cv::Mat GetBoundary(const cv::Mat marker, int w){
  cv::Mat boundarymap = cv::Mat::zeros(marker.rows,marker.cols, CV_8UC1);
  for(int r0 = 0; r0 < marker.rows; r0++){
    for(int c0 = 0; c0 < marker.cols; c0++){
      const int& i0 = marker.at<int>(r0,c0);
      bool b = false;
      for(int r1 = std::max(r0-w,0); r1 < std::min(r0+w,marker.rows); r1++){
        for(int c1 = std::max(c0-w,0); c1 < std::min(c0+w,marker.cols); c1++){
          const int& i1 = marker.at<int>(r1,c1);
          b = i0 != i1;
          if(b)
            break;
        }
        if(b)
          break;
      }
      if(!b)
        continue;
      boundarymap.at<unsigned char>(r0,c0) = true;
    }
  }
  return boundarymap;
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

  // Sum of diagonal elements in Hessian
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
py::array_t<unsigned char> PyGetBoundary(py::array_t<int32_t> _inputmask, int w) {
  cv::Mat inputmask = PyArray2Cv(_inputmask, CV_32SC1);
  cv::Mat boundary = GetBoundary(inputmask, w);
  py::array_t<unsigned char> output = Cv2PyArray<unsigned char>(boundary);
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

  // TODO BoNet ID notation과 호환성.
  const int min_label = 1;
  const int rows = buf_labels.shape[0];
  const int cols = buf_labels.shape[1];
  std::map<int,size_t> n_points;
  for (int r=0; r<rows; r++) {
    for (int c=0; c<cols; c++) {
        const int l = ((int32_t*) buf_labels.ptr)[r*cols+c];
        if(l < min_label)
          continue;
        n_points[l]++;
    }
  }

  std::map<int, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds;
  size_t sum = 0;
  for(auto it : n_points){
    int l = it.first;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
    ptr->reserve(it.second);
    clouds[l] = ptr;
    sum += it.second;
  }

  {
    const float min_depth = 0.0001;
    cv::Mat rgb(rows, cols, CV_8UC3);
    std::memcpy(rgb.data, buf_rgb.ptr, rows*cols*3*sizeof(unsigned char));
    std::vector<cv::Point2f> uv_points, normalized_points;
    std::vector<float> z_points;
    std::vector<cv::Scalar> colors;
    std::vector<int> l_points;
    uv_points.reserve(sum);
    z_points.reserve(sum);
    colors.reserve(sum);
    l_points.reserve(sum);

    for (int r=0; r<rows; r++) {
      for (int c=0; c<cols; c++) {
        const float& d = ((float*) buf_depth.ptr)[r*cols+c];
        if( d < min_depth)
          continue;
        const int l = ((int32_t*) buf_labels.ptr)[r*cols+c];
        if(l < min_label)
          continue;
        uv_points.push_back(cv::Point2f(c,r));
        z_points.push_back(d);
        const auto& pixel_rgb = rgb.at<cv::Vec3b>(r,c);
        cv::Scalar color(pixel_rgb[0],pixel_rgb[1],pixel_rgb[2]);
        colors.push_back(color);
        l_points.push_back(l);
      }
    }
    cv::undistortPoints(uv_points, normalized_points, K, D);
    for(size_t i = 0; i < normalized_points.size(); i++){
      const cv::Point2f& xbar = normalized_points.at(i);
      const float& z = z_points.at(i);
      const cv::Scalar& color = colors.at(i);
      const int l = l_points.at(i);
      pcl::PointXYZRGB pt;
      pt.x = xbar.x*z;
      pt.y = xbar.y*z;
      pt.z = z;
      pt.r = (float)color[0];
      pt.g = (float)color[1];
      pt.b = (float)color[2];
      auto ptr = clouds.at(l);
      ptr->push_back(pt);
    }
  }

  sum = 0;
  const float euclidean_tolerance = 10. * leaf_xy;
  for(auto it: clouds){
    auto ptr = it.second;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(ptr);
    sor.setLeafSize(leaf_xy, leaf_xy, leaf_z);
    sor.filter(*cloud_filtered);
    *ptr = *cloud_filtered;
    EuclideanCluster<pcl::PointXYZRGB>(euclidean_tolerance, ptr);
    sum += ptr->size();
  }

  const int n_point = sum;
  py::array_t<float> xyzrgb(6*n_point);
  py::array_t<int32_t> ins_points(n_point);
  xyzrgb.resize({n_point, 6});
  ins_points.resize({n_point,});
  py::buffer_info buf_ins_points = ins_points.request();

  float* xyzrgb_arr = (float*) xyzrgb.request().ptr;
  int32_t* ins_arr = (int32_t*) ins_points.request().ptr;
  {
    int i = 0; 
    for(auto it: clouds){
      int l= it.first;
      for(auto pt : *(it.second) ){
        float* ptr = xyzrgb_arr+6*i;
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
        ins_arr[i] = l;
        i++;
      }
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

py::tuple PyEvaluateEdgeDetection(py::array_t<int32_t> _gt_marker,
                                 py::array_t<int32_t> _pred_marker,
                                 py::array_t<unsigned char> _pred_outline,
                                 py::array_t<float> _depth,
                                 py::array_t<int32_t> _plane_marker,
                                 py::dict _plane2marker,
                                 py::dict _plane2normals,
                                 py::array_t<double> _K
                                 ){
  cv::Mat gt_marker = PyArray2Cv(_gt_marker, CV_32SC1);
  cv::Mat pred_marker = PyArray2Cv(_pred_marker, CV_32SC1);
  cv::Mat pred_outline = PyArray2Cv(_pred_outline, CV_8UC1);
  cv::Mat depth = PyArray2Cv(_depth, CV_32FC1);
  cv::Mat gt_outline = GetBoundary(gt_marker,1);
  cv::Mat plane_marker = PyArray2Cv(_plane_marker, CV_32SC1);
  std::map<int, int> plane2marker;
  EigenMap<int, Eigen::Vector4f> plane2coeff;
  for(auto it : _plane2marker)
    plane2marker[py::cast<int>(it.first)] = py::cast<int>(it.second);
  for(auto it : _plane2normals){
    py::array arr = py::cast<py::array>(it.second);
    double* ptr = (double*) arr.data();
    plane2coeff[py::cast<int>(it.first)] = Eigen::Vector4f(ptr[0],ptr[1],ptr[2],ptr[3]);
  }

  const int line_range = 5;
  cv::Mat thin_opposite; {
    cv::Mat dist_frombg;
    cv::Mat inverted_bg = gt_marker>0;
    inverted_bg.row(0) = 0;
    inverted_bg.row(gt_marker.rows-1) = 0;
    inverted_bg.col(0) = 0;
    inverted_bg.col(gt_marker.cols-1) = 0;
    cv::distanceTransform(inverted_bg, dist_frombg, cv::DIST_L2, cv::DIST_MASK_3);

    // 가장 가까운 건너편 segment를 어떻게 찾고, 연결하느냐 문제.
    GetNeighbors(plane_marker,line_range, plane2marker, false, thin_opposite);
    for(int r=0; r<plane_marker.rows;r++){
      for(int c=0; c<plane_marker.cols;c++){
        int& pidx1 = thin_opposite.at<int>(r,c);
        if(dist_frombg.at<float>(r,c) < 15.) // Near background with possible depth noise
          pidx1 = 0;
      }
    }
  }

  Eigen::Matrix<float,3,3> K = PyArray2Eigen<double,3,3>(_K).cast<float>();
  Eigen::Matrix<float,3,3> invK = K.inverse();
  const int rows = gt_marker.rows;
  const int cols = gt_marker.cols;
  /*
  Ground truth boundary 상에서 
  1. Neighbor pair별,
    1-1. boundary points 및 정보. dict[(ins0,ins1)] = Nxm points array
      b. done) array(1,bool) : dist_trans(pred_boundary) < threshold[pixel] . 즉, Outline Recall 여부.
      c. done) array(1,float) : 깊이값
      *. done) array(1,float) : 두 instance 사이 normal 각 차이 (python에서)
    1-2. Segmentation 성공여부
      * dist_trans(boundary) < threshold 범위 내에서 
      * gi0, gi1 각각에 가장 많이 겹치는 pred id 가 서로 다르면 segmentation 성공, 같으면 실패.
  */
  cv::Mat dist_predoutline, dist_gtoutline;
  cv::distanceTransform(pred_outline<1, dist_predoutline, cv::DIST_L2, cv::DIST_MASK_3);
  cv::distanceTransform(gt_outline<1, dist_gtoutline, cv::DIST_L2, cv::DIST_MASK_3);
  cv::Mat dst = GetColoredLabel(plane_marker,true);

  struct BoundaryStat{
    cv::Point2i pt;
    bool pred_detection;
    float depth;
    float oblique;
    float plane_offset;
  };

  std::map<int,size_t> gt_areas;
  for(int r=0; r<rows;r++){
    for(int c=0; c<cols;c++){
      const int& m0 = gt_marker.at<int>(r,c);
      if(m0 == 0)
        continue;
      gt_areas[m0]++;
    }
  }

  std::map<std::pair<int,int>, std::list<BoundaryStat> > boundary_stats;
  std::map<std::pair<int,int>, int> boundary_n, boundary_recall;
  std::map<std::pair<int,int>, bool> boundary_segmentation;
  int n_boundaries = 0;
  const float fline_range = line_range; 
  for(int r=0; r<rows;r++){
    for(int c=0; c<cols;c++){
      const float& z = depth.at<float>(r,c);
      if(z < .001)
        continue;
      if(thin_opposite.at<int>(r,c)==0)
        continue;
      const int& pidx0 = plane_marker.at<int>(r,c);
      const int& pidx1 = thin_opposite.at<int>(r,c);
      if(!plane2coeff.count(pidx0))
        continue;
      if(!plane2coeff.count(pidx1))
        continue;
      if(!plane2marker.count(pidx0))
         continue;
      if(!plane2marker.count(pidx1))
         continue;
      const auto& n0 = plane2coeff.at(pidx0).head<3>();
      const auto& n1 = plane2coeff.at(pidx1).head<3>();
      Eigen::Vector4f pt0(0.,0.,0.,1.);
      pt0.head<3>() = z * invK * Eigen::Vector3f( (float)c, (float)r, 1.);
      const float err_self = std::abs( plane2coeff.at(pidx0).dot(pt0) );
      if(err_self > .002)
        continue;
      BoundaryStat stat;
      stat.pt = cv::Point2i(c,r);
      stat.pred_detection = dist_predoutline.at<float>(r,c) < fline_range;
      stat.depth = z;
      stat.oblique = 180. / M_PI * std::acos( n0.dot(n1) );
      stat.plane_offset = std::abs( plane2coeff.at(pidx1).dot(pt0) );
      if(stat.plane_offset > 0.1)
        cv::circle(dst, cv::Point2i(c,r), 2,CV_RGB(0,0,0), -1 );
      const int& m0 = plane2marker.at(pidx0);
      const int& m1 = plane2marker.at(pidx1);
      const auto key = std::make_pair(std::min(m0,m1),std::max(m0,m1));
      if(!boundary_n.count(key)){
        boundary_n[key] = 0;
        boundary_recall[key] = 0;
        boundary_segmentation[key] = false;
      }
      boundary_stats[key].push_back(stat);
      boundary_n[key]++;
      if(stat.pred_detection)
        boundary_recall[key]++;
      n_boundaries++;
    }
  }

  {
    std::map<int,std::map<int,size_t> > gt2pred_counts;
    for(int r=0; r<rows;r++){
      for(int c=0; c<cols;c++){
        const int& gt = gt_marker.at<int>(r,c);
        if(gt < 1)
          continue;
        const int& pred = pred_marker.at<int>(r,c);
        if(pred < 1)
          continue;
        gt2pred_counts[gt][pred]++;
      }
    } // for r

    std::map<int,int> gt2pred;
    std::map<int, std::set<int> > pred2gt;
    for(const auto& it : gt2pred_counts){
      const int& gt = it.first;
      int max_pred = -1;
      size_t max_count = .5*gt_areas.at(gt);
      for(const auto& it1 : it.second){
        if(it1.second < max_count)
          continue;
        max_pred = it1.first;
        max_count = it1.second;
      }
      gt2pred[gt] = max_pred;
      pred2gt[max_pred].insert(gt);
    }

    for(auto& it : boundary_segmentation){
      const int& m0 = it.first.first;
      const int& m1 = it.first.second;
      if(!gt2pred.count(m0))
        continue;
      if(!gt2pred.count(m1))
        continue;
      it.second = gt2pred[m0] != gt2pred[m1];
      //printf("boundary (%d,%d)'s segmentation = %d\n", m0,m1,it.second);
    }
  }

  py::list pyoversegment_stats;
  {
    std::map<int,std::map<int,size_t> > gt2pred_counts;
    for(int r=0; r<rows;r++){
      for(int c=0; c<cols;c++){
        const float& d = dist_gtoutline.at<float>(r,c);
        if(d < 2.)
          continue;
        const int& gt = gt_marker.at<int>(r,c);
        if(gt < 1)
          continue;
        const int& pred = pred_marker.at<int>(r,c);
        if(pred < 1) // Ignore bg 
          continue;
        gt2pred_counts[gt][pred]++;
      }
    }

    //std::set<int> oversegmented_instances;
    for(const auto& it : gt2pred_counts){
      const int& m0 = it.first;
      bool oversegmented = false;
      std::vector< std::pair<int,size_t> > counts;
      size_t sum = 0;
      for(const auto& it2 : it.second){
        sum += it2.second;
        counts.push_back(std::make_pair(it2.first, it2.second) );
      }
      std::sort(counts.begin(), counts.end(), [](const std::pair<int,size_t>& a,
                                                 const std::pair<int,size_t>& b) {return a.second > b.second; } );
      if(counts.size() > 1){
        const float n1 = counts[0].second;
        const float n2 = counts[1].second;
        oversegmented = n2 > .2 * n1;
      }
      //if(oversegmented)
      //  oversegmented_instances.insert(m0);
      py::tuple pystat = py::make_tuple(m0,oversegmented);
      pyoversegment_stats.append(pystat);
    }
  }

  {
    //cv::imshow("dist_gtoutline", 0.01*dist_gtoutline);
    //cv::imshow("pred_detection", GetColoredLabel(pred_marker,true) );
    //cv::imshow("marker", GetColoredLabel(gt_marker,true) );
    //cv::imshow("plane_marker", dst);
    //cv::imshow("thin_opposite", GetColoredLabel(thin_opposite) );
    //cv::waitKey(0);
  }

  py::list pyboundary_stats;
  py::list pyboundary_recall_segment;
  for(const auto& it : boundary_stats){
    const int& m0 = it.first.first;
    const int& m1 = it.first.second;
    const std::list<BoundaryStat> stats = it.second;
    BoundaryStat rep_stat; // Representative stats
    rep_stat.oblique = -99999.;
    rep_stat.plane_offset = -99999.;
    rep_stat.depth = -99999.;
    std::vector<double> plane_offsets;
    plane_offsets.reserve(stats.size());
    for(const auto& stat : stats){
      rep_stat.oblique = std::max(rep_stat.oblique, stat.oblique);
      //rep_stat.plane_offset = std::max(rep_stat.plane_offset, stat.plane_offset);
      plane_offsets.push_back(stat.plane_offset);
      rep_stat.depth = std::max(rep_stat.depth,stat.depth);
      // TODO boundary id?
      py::tuple pystat
        = py::make_tuple(stat.pred_detection,stat.depth,stat.oblique,stat.plane_offset);
      pyboundary_stats.append(pystat);
    }
    std::sort(plane_offsets.begin(), plane_offsets.end());
    rep_stat.plane_offset = plane_offsets[plane_offsets.size()/2];
    float recall = (float)boundary_recall.at(it.first) / (float) boundary_n.at(it.first);
    bool bseg = boundary_segmentation.at(it.first);
    pyboundary_recall_segment.append( py::make_tuple(recall,
                                                     bseg,
                                                     m0,
                                                     m1,
                                                     rep_stat.depth,
                                                     rep_stat.oblique,
                                                     rep_stat.plane_offset) );
  }
  return py::make_tuple(pyboundary_stats, pyboundary_recall_segment, pyoversegment_stats);
}

py::dict PyGetNeighbors(py::array_t<int32_t> _marker, int radius) {
  py::dict output;
  //py::buffer_info buf_marker = marker.request();
  //const int32_t* ptr_marker = (const int32_t*) buf_marker.ptr;
  cv::Mat marker = PyArray2Cv(_marker, CV_32SC1);
  std::set<std::pair<int,int> > contacts = GetNeighbors(marker, radius);
  for(const auto& contact : contacts){
    py::object i0 = py::cast<int>((int)contact.first);
    py::object i1 = py::cast<int>((int)contact.second);

    if(! output.contains(i0) )
      output[i0] = py::list();
    py::cast<py::list>( output[i0] ).append(i1);

    if(! output.contains(i1) )
      output[i1] = py::list();
    py::cast<py::list>( output[i1] ).append(i0);

  }
  return output;
}

py::tuple PyComputeOBB(py::array_t<int32_t> _frontmarker,
                       py::array_t<int32_t> _marker,
                       py::list py_label2vertices,
                       py::array_t<float> _depth,
                       py::array_t<float> _numap,
                       py::array_t<float> _nvmap,
                       float max_depth,
                       py::array_t<int32_t> _plane_marker,
                       py::dict _plane2marker,
                       py::dict _plane2centers
                       ){
  cv::Mat depth = PyArray2Cv<float>(_depth,CV_32FC1);
  cv::Mat marker = PyArray2Cv<int32_t>(_marker, CV_32SC1);
  cv::Mat front_marker = PyArray2Cv<int32_t>(_frontmarker, CV_32SC1);
  cv::Mat numap = PyArray2Cv<float>(_numap, CV_32FC1);
  cv::Mat nvmap = PyArray2Cv<float>(_nvmap, CV_32FC1);
  cv::Mat plane_marker = PyArray2Cv<int32_t>(_plane_marker, CV_32SC1);
  std::map<int, int> plane2marker;
  std::map<int, cv::Point2i>  plane2center;
  for(auto it : _plane2marker){
    int pidx = py::cast<int>(it.first);
    int midx = py::cast<int>(it.second);
    plane2marker[pidx] = midx;
  }
  for(auto it : _plane2centers){
    int pidx = py::cast<int>(it.first);
    py::tuple xy = py::cast<py::tuple>(it.second);
    int x = py::cast<int>(xy[0]);
    int y = py::cast<int>(xy[1]);
    plane2center[pidx] = cv::Point2i(x,y);
  }

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
  std::map<int, OBB> obbs = ComputeOBB(front_marker,
                                         marker,
                                         depth,
                                         label2vertices,
                                         numap,
                                         nvmap,
                                         max_depth
                                         );
  EigenMap<int, EigenVector<Eigen::Vector4f> > front_side_planes;
  std::map<int, std::vector<int> > inliers;
  //std::cout << "output =(";
  for(const auto& it : obbs){
    const int& midx = it.first;
    front_side_planes[midx] = GetFrontSidePlanes(it.second);
    //std::cout << it.first << ",";
  }
  //std::cout << ")\n";

  for(int r=0; r<plane_marker.rows; r++){
    for(int c=0; c<plane_marker.cols; c++){
      const int& pidx = plane_marker.at<int>(r,c);
      if(!plane2marker.count(pidx))
        continue;
      const int& midx = plane2marker.at(pidx);
      if(!front_side_planes.count(midx) )
        continue;
      const cv::Point2i cp = plane2center.at(pidx);
      if(front_marker.at<int>(cp.y,cp.x) == midx){
          //std::cout << "pidx #" << pidx << " is front plane" << std::endl;
        continue;
      }
      float nu = numap.at<float>(r,c);
      float nv = nvmap.at<float>(r,c);
      const float& d = depth.at<float>(r,c);
      const auto pt = Eigen::Vector4f(nu*d,nv*d,d, 1.);
      int opt_k = -1;
      float min_err = 99999.;
      for(int k = 1; k < 5; k++){
        const Eigen::Vector4f& plane = front_side_planes.at(midx).at(k);
        float err = std::abs( plane.dot(pt) );
        if(err > min_err)
          continue;
        min_err = err;
        opt_k = k;
      }
      if(opt_k < 0)
        continue;
      if(!inliers.count(pidx))
        inliers[pidx].resize(5);
      inliers[pidx][opt_k]++;
    }
  }

  //for(const auto& it : inliers){
  //  std::cout << "Counts in pidx#" << it.first << " : ";
  //  for(const int& c : it.second)
  //    std::cout << c << ", ";
  //  std::cout << std::endl;
  //}

  py::list obb_list;
  for(auto it : obbs){
    const OBB& obb = it.second;
    py::tuple pose = py::make_tuple(obb.Tcb_xyz_qwxyz_[0], obb.Tcb_xyz_qwxyz_[1],
                                    obb.Tcb_xyz_qwxyz_[2], obb.Tcb_xyz_qwxyz_[3],
                                    obb.Tcb_xyz_qwxyz_[4], obb.Tcb_xyz_qwxyz_[5],
                                    obb.Tcb_xyz_qwxyz_[6]);
    py::tuple scale = py::make_tuple(obb.scale_[0], obb.scale_[1], obb.scale_[2]);
    obb_list.append(py::make_tuple(it.first, pose, scale) );
  }
  py::dict plane2normals;
  std::set<int> sides;
  for(auto it : inliers){
    int pidx = it.first;
    const int& midx = plane2marker.at(pidx);
    const std::vector<int>& counts = it.second;
    int opt_k = -1;
    int max_count = 0;
    for(int k = 1; k < 5; k++){
      if(max_count > counts.at(k))
        continue;
      max_count = counts.at(k);
      opt_k = k;
    }
    if(opt_k < 0)
      continue;
    //std::cout << "side plane pidx#" << pidx << ", k#" << opt_k << ", counts=" << max_count << std::endl;
    sides.insert(pidx);
    const Eigen::Vector4f& plane = front_side_planes.at(midx).at(opt_k);
    plane2normals[py::cast(pidx)] = py::make_tuple(plane[0],plane[1],plane[2],plane[3]);
  }
  for(auto it : plane2marker){
    const int& pidx = it.first;
    const int& midx = it.second;
    if(sides.count(pidx))
      continue;
    const Eigen::Vector4f& plane = front_side_planes.at(midx).at(0);
    plane2normals[py::cast(pidx)] = py::make_tuple(plane[0],plane[1],plane[2],plane[3]);
  }
  return py::make_tuple(obb_list, plane2normals);
}


PYBIND11_MODULE(unet_ext, m) {
  m.def("EvaluateEdgeDetection", &PyEvaluateEdgeDetection, "Evaluate edge detection",
        py::arg("gt_marker"),py::arg("pred_marker"),py::arg("pred_outline"), py::arg("depth"),
        py::arg("plane_marker"), py::arg("plane2marker"), py::arg("plane2normals"),
        py::arg("camera_marix"));
  m.def("GetNeighbors", &PyGetNeighbors, "Get neighbors of each instance in a given marker.",
        py::arg("marker"), py::arg("radius") );
  m.def("GetFilteredDepth", &PyGetFilteredDepth, "Get filtered depth.",
        py::arg("input_mask"), py::arg("dd_edge"),
        py::arg("sample_width") );
  m.def("GetBoundary", &PyGetBoundary, "find edge", py::arg("input_mask"), py::arg("width") );
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

  m.def("ComputeOBB", &PyComputeOBB, "Compute OBB from given marker and depth map",
        py::arg("front_marker"),
        py::arg("marker"),
        py::arg("label2vertices"),
        py::arg("depth"),
        py::arg("nu_map"),
        py::arg("nv_map"),
        py::arg("max_depth"),
        py::arg("plane_marker"), py::arg("plane2marker"), py::arg("plane2centers")
        );
}
