#include "ros_util.h"
#include "ros_unet/GetBg.h"
#include "ros_unet/Plane.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/Imu.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen3/Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
//#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>

//#include <g2o/types/slam3d/se3quat.h>
#include <chrono>

void cvtCameraInfo2CvMat(const sensor_msgs::CameraInfo& cam_info, cv::Mat& K, cv::Mat& D){
  K = (cv::Mat_<double>(3, 3) << cam_info.K[0], 0, cam_info.K[2],
               0, cam_info.K[4], cam_info.K[5],
               0,  0,  1.);
  D = cv::Mat::zeros(cam_info.D.size(), 1, CV_64F);
  for(size_t i = 0; i < D.rows; i++)
    D.at<double>(i,0) = cam_info.D[i];
  return;
}

void Unproject(cv::Mat depth,
               cv::Mat nu_map,
               cv::Mat nv_map,
               cv::Mat init_mask,
               int step,
               pcl::PointCloud<pcl::PointXYZL>::Ptr& cloud,
               std::vector<cv::Point2i>& uvs
              ){
  cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
  int n = cv::countNonZero(init_mask);
  cloud->reserve(n);
  uvs.reserve(n);
  for(int r=0; r<depth.rows; r+=step){
    for(int c=0; c<depth.cols; c+=step){
      const float& z = depth.at<float>(r,c);
      if(z < 0.001)
        continue;
      if( init_mask.at<unsigned char>(r,c) <1 )
        continue;
      const float& nu = nu_map.at<float>(r,c);
      const float& nv = nv_map.at<float>(r,c);
      pcl::PointXYZL pt;
      pt.x = nu*z; pt.y = nv*z; pt.z = z; pt.label = 0;
      cloud->points.push_back(pt);
      uvs.push_back(cv::Point2i(c,r));
    }
  }
  return;
}

void regionGrowing(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
                   std::map<u_int32_t, pcl::PointCloud<pcl::PointXYZL>::Ptr >& clusters //, std::map<u_int32_t, pcl::PointXYZL>& rg_means
                   ){
  pcl::search::Search<pcl::PointXYZL>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZL>);
  pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZL, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setKSearch(10);
  normal_estimator.compute(*normals);

  /*
    * Refernce : https://pointclouds.org/documentation/classpcl_1_1_region_growing.html
    * Param
      * NumberOfNeighbours : KNN의 neighbor 숫자.
      * Curvature(default:1.) : Curvature가 threshold 이하일 경우, plane으로 취급안함.
      * Smoothness            : 이웃한 point와 normal vector의 오차 허용범위.

  */
  pcl::RegionGrowing<pcl::PointXYZL, pcl::Normal> reg;
  reg.setSearchMethod(tree);
  reg.setInputCloud(cloud);
  reg.setInputNormals(normals);
  reg.setMinClusterSize(50);
  //reg.setIndices(indices);
  //reg.setMaxClusterSize(1000000);
  reg.setNumberOfNeighbours(20);
  reg.setSmoothnessThreshold(5.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(.1); // .1

  std::vector<pcl::PointIndices> _clusters;
  reg.extract(_clusters);

  const float max_r2 = 3.*3.;
  for(u_int32_t i =0; i<_clusters.size();i++){
    const auto& indices = _clusters.at(i).indices;
    std::vector<float> vr;
    vr.reserve(indices.size());
    for(const int& j : indices){
      const auto& pt0 = cloud->at(j);
      vr.push_back(pt0.x*pt0.x+pt0.y*pt0.y+pt0.z*pt0.z);
    }
    std::sort(vr.begin(), vr.end());
    pcl::PointCloud<pcl::PointXYZL>::Ptr g(new pcl::PointCloud<pcl::PointXYZL>());
    g->reserve(indices.size());
    u_int32_t label = i+1;
    if(*vr.begin() > max_r2)
      label = 0;
    for(const int& j : indices){
      auto& pt0 = cloud->at(j);
      pt0.label = label;
      g->push_back(pt0);
    }
#if 1
    if(label < 1 && clusters.count(0) )
      *clusters[0] += *g;
    else
      clusters[label] = g;
#else
#endif
  }
  return;
}

void getPlanes(std::map<u_int32_t, pcl::PointCloud<pcl::PointXYZL>::Ptr >& clusters,
              EigenMap<u_int32_t, Eigen::Vector4f>& p_coeffs
              ) {
  for(auto it : clusters){
    const int& label = it.first;
    if(label < 1)
      continue;
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = it.second;
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZL> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Optional
    seg.setOptimizeCoefficients(true);

    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.04); // TODO Median?
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    Eigen::Vector4f coeff;
    if(coefficients->values[3] < 0.)
      coeff = Eigen::Vector4f(-coefficients->values[0],
                              -coefficients->values[1],
                              -coefficients->values[2],
                              -coefficients->values[3]);

    else
      coeff = Eigen::Vector4f(coefficients->values[0],
                              coefficients->values[1],
                              coefficients->values[2],
                              coefficients->values[3]);
    p_coeffs[label] = coeff;
  }

  return;
}

size_t getNumberOfPointsBehindPlane(pcl::search::Search<pcl::PointXYZL>::Ptr sparse_tree,
                                    const std::map<u_int32_t, pcl::PointCloud<pcl::PointXYZL>::Ptr >& p_clusters,
                                    pcl::PointCloud<pcl::PointXYZL>::ConstPtr plane_cloud,
                                    const Eigen::Vector4f& coeff,
                                    u_int32_t plane_label,
                                    float distance_th){
  pcl::PointCloud<pcl::PointXYZL>::ConstPtr sparse_cloud = sparse_tree->getInputCloud();
  const float radius = 1.; // [meter]
  std::vector<int> indices;
  std::vector<float> sqr_distances;
  if(distance_th < 0.)
    ROS_WARN_STREAM("distance_th should be positive");

  size_t n = 0;
  for(const auto& pt0 : *plane_cloud){
    sparse_tree->radiusSearch(pt0, radius, indices, sqr_distances);
    for(const int& i : indices){
      const auto& pt1 = sparse_cloud->at(i);
      if(pt1.label==plane_label)
        continue;
      if(pt1.label < 1)
        continue;
      float d = coeff.dot(Eigen::Vector4f(pt1.x,pt1.y,pt1.z,1.));
      if(d + distance_th < 0.)
        n++;
    }
  }
  return n;
}

#if 0
void getFloorAndWall(std::map<u_int32_t, pcl::PointCloud<pcl::PointXYZL>::Ptr > p_clusters,
                     pcl::search::Search<pcl::PointXYZL>::Ptr p_tree,
                     const EigenMap<u_int32_t, Eigen::Vector4f>& p_coeffs,
                     const cv::Mat marker,
                     const Eigen::Vector3f& floor_norm_prediction,
                     bool given_imu,
                     const Eigen::Vector3d& linear_acc,
                     std::vector<int>& l_walls,
                     cv::Mat& mask
                       ){
  /*
    outermost_plane 중에서 l_floor, l_ceiling 찾아내는게 목적.
  */
  const float perpen_th = std::sin(M_PI/180.*80.); // 90deg에 가까울수록 엄밀한,
  const float voxel_leaf = 0.1;
  std::set<int> outermost_planes;
  for(auto it_p : p_clusters){
    const u_int32_t& l = it_p.first;
    if(l < 1) // Ignore too far outer points
      continue;
    const Eigen::Vector4f& p = p_coeffs.at(l);
    std::vector<float> depths;
    depths.reserve(it_p.second->size());
    for(const auto& pt : *it_p.second)
      depths.push_back(pt.z);
    std::sort(depths.begin(),depths.end());
    const float& median = depths.at(depths.size()/2);
    const float distance_th = .1*median;
    // TODO Range 또는 angle 제한 필요.
    int n = 0.2 * it_p.second->size() ;
    bool is_outermost = n > getNumberOfPointsBehindPlane(p_tree,p_clusters,it_p.second,p,l,distance_th);
    if(is_outermost)
      outermost_planes.insert(l);
  }
  const float distance_th = .15;
  int l_floor = -1;
  const float th_floor = given_imu?std::cos(M_PI/180.*20.):std::cos(M_PI/180.*40.);
  { // Search floor
    std::vector<std::pair<int, size_t> > candidates;
    for(const auto& l : outermost_planes){
      const Eigen::Vector4f& p = p_coeffs.at(l);
      if(floor_norm_prediction.dot(p.head<3>()) < th_floor)
        continue;
      candidates.push_back(std::make_pair(l,p_clusters.at(l)->size()));
    }
    std::sort(std::begin(candidates),
              std::end(candidates),
              [](const auto& a, const auto& b) {
              return a.second > b.second; });
    if(!candidates.empty()){
      l_floor = candidates.front().first;
      std::vector<cv::Point> locations;
      cv::findNonZero(marker == l_floor, locations);
      for (const auto& pt : locations)
        mask.at<int32_t>(pt) = 1;  // FLOOR 1
    }
  }

  for(const auto&l : outermost_planes){
    if(l==l_floor)
      continue;
    if(l_floor > 0){ // TODO perpendicular constraint 적용하고도 bg wall 인식되야함.
      if(!p_coeffs.count(l_floor))
        continue;
      if(!p_coeffs.count(l))
        continue;
      const Eigen::Vector4f& p0 = p_coeffs.at(l_floor);
      const Eigen::Vector4f& p = p_coeffs.at(l);
      // Perpendicular wall
      bool not_perpen = p0.head<3>().cross(p.head<3>()).norm() < perpen_th;
      // Ceil
      bool not_ceil = p0.head<3>().dot(p.head<3>()) > -th_floor;
      if( not_perpen && not_ceil)
        continue;
    }
    std::vector<cv::Point> locations;
    cv::findNonZero(marker == l, locations);
    for (const auto& pt : locations)
      mask.at<int32_t>(pt) = 2;  // OTHER_WALL 2
  }

  for(int r=0; r<marker.rows; r++)
    for(int c=0; c<marker.cols; c++)
      if(marker.at<int32_t>(r,c) < 1)
        mask.at<int32_t>(r,c) = 3; // Non flat
  return;
}

void EuclideanFilter(boost::shared_ptr<pcl::search::KdTree<pcl::PointXYZL> > sparse_tree,
                  float square_tolerance,
                  std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr >& dense_clusters,
                  pcl::PointCloud<pcl::PointXYZL>::Ptr& dense_cloud
                  ){
  pcl::PointCloud<pcl::PointXYZL>::ConstPtr sparse_cloud = sparse_tree->getInputCloud();
  pcl::PointIndices::Ptr results(new pcl::PointIndices);
  // Creating the KdTree object for the search method of the extraction
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZL> ec;
  ec.setClusterTolerance(square_tolerance);
  ec.setMinClusterSize(100);
  ec.setSearchMethod(sparse_tree);
  ec.setInputCloud(sparse_cloud);
  ec.extract(cluster_indices);
  if(cluster_indices.empty())
    return;
  std::sort(std::begin(cluster_indices),
            std::end(cluster_indices),
            [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() > b.indices.size(); });
  const bool reserve_best_only = true;
  int n = 0;
  for(const pcl::PointIndices& indices : cluster_indices){
    n += indices.indices.size();
    if(reserve_best_only)
      break;
  }
  results->indices.reserve(n);
  for(const pcl::PointIndices& indices : cluster_indices){
    for(const int& index : indices.indices)
      results->indices.push_back(index);
    if(reserve_best_only)
      break;
  }

  std::set<int> sparse_inliers_indices(results->indices.begin(),results->indices.end());
  std::set<int> outlier_labels;
  std::vector<int> neighbors(1);
  std::vector<float> square_distance(1);
  size_t n_outlier = 0;
  for(pcl::PointXYZL& dense_pt : *dense_cloud){
    sparse_tree->nearestKSearch(dense_pt, 1, neighbors, square_distance);
    if(sparse_inliers_indices.count(neighbors.at(0)))
      continue;
    // Euclidean filter 결과 outlier
    outlier_labels.insert(dense_pt.label);
    dense_pt.label = 0;
    n_outlier++;
  }
  pcl::PointCloud<pcl::PointXYZL>::Ptr tmp_inliers(new pcl::PointCloud<pcl::PointXYZL>);
  pcl::PointCloud<pcl::PointXYZL>::Ptr tmp_outliers(new pcl::PointCloud<pcl::PointXYZL>);
  tmp_inliers->reserve(dense_cloud->size()-n_outlier);
  tmp_outliers->reserve(n_outlier);
  for(pcl::PointXYZL& dense_pt : *dense_cloud){
    if(dense_pt.label>0)
      tmp_inliers->push_back(dense_pt);
    else
      tmp_outliers->push_back(dense_pt);
  }
  dense_cloud = tmp_inliers;
  for(const int& l : outlier_labels)
    dense_clusters.erase(l);
  dense_clusters[0] = tmp_outliers;
  return;
}


#endif

void projectClusters(pcl::PointCloud<pcl::PointXYZL>::Ptr merged_p_cloud,
                     pcl::search::Search<pcl::PointXYZL>::Ptr merged_p_tree,
                     pcl::PointCloud<pcl::PointXYZL>::Ptr dense_cloud,
                     int step,
                     const std::vector<cv::Point2i>& uvs,
                     cv::Mat& p_marker//, cv::Mat& p_mask
                    ){
  float max_square_tolerance = .4; // [meter^2] TODO
  max_square_tolerance *= max_square_tolerance;

  std::vector<int> neighbors(1);
  std::vector<float> square_distance(1);
  for(int i=0; i<uvs.size(); i++){
    const cv::Point2i& uv = uvs.at(i);
    pcl::PointXYZL& pt = dense_cloud->at(i);
    if(pt.z > 10.)
      continue;
    merged_p_tree->nearestKSearch(pt, 1, neighbors, square_distance);
    if(square_distance.empty())
      continue;
    if(square_distance.at(0) > max_square_tolerance)
      continue;
    const pcl::PointXYZL& pt0 = merged_p_cloud->at(neighbors.at(0));
#if 1
    int label = pt0.label > 0? pt0.label : -1; // Labeling cloud
#else
    int label = pt0.label; // Labeling cloud
    if(label < 1){ // pcl::PointXYZL.label은 uint, mask는 int
      continue;
#endif
    cv::rectangle(p_marker,cv::Rect(uv,uv+cv::Point2i(step,step)),label);
  }
  return;
}

void getBgMask(std::map<u_int32_t, pcl::PointCloud<pcl::PointXYZL>::Ptr > p_clusters,
                  const EigenMap<u_int32_t, Eigen::Vector4f>& p_coeffs,
                  pcl::search::Search<pcl::PointXYZL>::Ptr p_tree,
                  const cv::Mat& plane_marker,
                  const Eigen::Vector3f linear_acc,
                  cv::Mat& bg_mask
                 ){
  Eigen::Vector3f gravity_dir = -linear_acc.normalized();
  bool look_upper = gravity_dir.z() < 0.;
  //ROS_INFO_STREAM("gdir="<<gravity_dir.transpose());
  assert(look_upper); // TODO Test with look floor

  std::set<u_int32_t> outermost_planes;
  for(auto it_p : p_clusters){
    const u_int32_t& l = it_p.first;
    if(!p_coeffs.count(l))
      continue;
    const Eigen::Vector4f& p = p_coeffs.at(l);
    if(look_upper && gravity_dir.dot(p.head<3>()) < -.7) // No floor when look upper at testbed
      continue;
    if( (!look_upper) && gravity_dir.dot(p.head<3>()) >.7)
      continue;

    std::vector<float> depths;
    depths.reserve(it_p.second->size());
    for(const auto& pt : *it_p.second)
      depths.push_back(pt.z);
    std::sort(depths.begin(),depths.end());
    const float& median = depths.at(depths.size()/2);

    float distance_th = .1*median;
    if(gravity_dir.dot(p.head<3>()) > .7)
      distance_th = .2*median; // Bended ceil of testbed.

    int n = 0.2 * it_p.second->size() ;
    bool is_outermost = n > getNumberOfPointsBehindPlane(p_tree,
                                                         p_clusters,
                                                         it_p.second,
                                                         p,l,distance_th);
    if(is_outermost)
      outermost_planes.insert(l);
  }

  for(const auto&l : outermost_planes){
    std::vector<cv::Point> locations;
    cv::findNonZero(plane_marker == l, locations);
    for (const auto& pt : locations)
      bg_mask.at<int32_t>(pt) = 1;  // Wall
  }
  {
    std::vector<cv::Point> locations;
    cv::findNonZero(plane_marker == -1, locations);
    for (const auto& pt : locations)
      bg_mask.at<int32_t>(pt) = 2;  // Too far or none plane, outliers
  }

  return;
}

class BgDetector {
public:
  BgDetector(ros::NodeHandle nh) :
    nh_(nh),
    rgb_sub_(nh.subscribe<sensor_msgs::Image>("rgb", 1, &BgDetector::rgbCallback, this)),
    depth_sub_(nh.subscribe<sensor_msgs::Image>("depth", 1, &BgDetector::depthCallback, this)),
    camera_info_sub_(nh.subscribe<sensor_msgs::CameraInfo>("info", 1, &BgDetector::cameraInfoCallback, this)),
    linear_acc_(new Eigen::Vector3d(0,-1.,0.))
  {
    pub_points_ = nh_.advertise<sensor_msgs::PointCloud2>("points", 1);
    pub_rg_ = nh_.advertise<sensor_msgs::PointCloud2>("rg", 1);
    pub_imu_ = nh_.advertise<sensor_msgs::Imu>("output_imu",1);
    pub_planeseg_ = nh.advertise<sensor_msgs::Image>("plane_seg",1);
    pub_bgmask_ = nh.advertise<sensor_msgs::Image>("bg_mask",1);
  }
  ~BgDetector(){
    delete linear_acc_;
  }

  bool GetBg(ros_unet::GetBg::Request& req,
             ros_unet::GetBg::Response& res){
    { sensor_msgs::ImageConstPtr ptr = boost::make_shared<const sensor_msgs::Image>(req.rgb);
      rgbCallback(ptr);
    }
    { sensor_msgs::ImageConstPtr ptr = boost::make_shared<const sensor_msgs::Image>(req.depth);
      depthCallback(ptr);
    }
    { sensor_msgs::CameraInfoConstPtr ptr = boost::make_shared<const sensor_msgs::CameraInfo>(req.info);
      cameraInfoCallback(ptr);
    }
    { sensor_msgs::ImuConstPtr ptr = boost::make_shared<const sensor_msgs::Imu>(req.imu);
      imuCallback(ptr);
    }
    /*
    ros::Rate rate(10);
    while(linear_acc_->norm() == 0.) {
      rate.sleep();
      ros::spinOnce();
    }
    */
    bool r = Process(res);
    return r;
  }

  bool Process(ros_unet::GetBg::Response& res){
    if(rgb_.empty())
      return false;
    if(depth_.empty())
      return false;
    if(nu_map_.empty())
      return false;
    /*
    if(linear_acc_->norm() == 0.)
      return false;
    */
    cv::Mat init_mask = cv::Mat::ones(depth_.rows,depth_.cols,CV_8UC1);

    pcl::PointCloud<pcl::PointXYZL>::Ptr dense_cloud; // the dense cloud
    std::vector<cv::Point2i> uvs;
    int step = 2;
    Unproject(depth_, nu_map_, nv_map_, init_mask, step, dense_cloud, uvs);
    pcl::PointCloud<pcl::PointXYZL>::Ptr voxel_cloud;
    {
      voxel_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      const float voxel_leaf=.02;
      pcl::VoxelGrid<pcl::PointXYZL> sor;
      sor.setInputCloud(dense_cloud);
      sor.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
      sor.filter(*voxel_cloud);
      for(auto& pt : *voxel_cloud)
        pt.label = 0; // TODO should be 0, but bug..
    }

    std::map<u_int32_t, pcl::PointCloud<pcl::PointXYZL>::Ptr > rg_clusters;
    //std::map<u_int32_t, pcl::PointXYZL> rg_means;
    regionGrowing(voxel_cloud, rg_clusters);

    EigenMap<u_int32_t, Eigen::Vector4f> p_coeffs;
    getPlanes(rg_clusters, p_coeffs);

    if(pub_rg_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      //std::map<int,pcl::PointCloud<pcl::PointXYZL>::Ptr> dst;
      //if(rg_clusters.count(0))
      //  dst[0] = rg_clusters.at(0);
      //ColorizeSegmentation(dst, msg);
      ColorizeSegmentation(rg_clusters, msg);
      msg.header.frame_id = frame_id_;
      pub_rg_.publish(msg);
    }

    // Mapping bg plane on 2D image
    pcl::PointCloud<pcl::PointXYZL>::Ptr merged_p_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::search::Search<pcl::PointXYZL>::Ptr merged_p_tree(new pcl::search::KdTree<pcl::PointXYZL>);
    cv::Mat plane_marker = cv::Mat::zeros(depth_.rows,depth_.cols, CV_32SC1);
    cv::Mat bg_mask = cv::Mat::zeros(depth_.rows, depth_.cols, CV_32SC1);
    {
      pcl::PointCloud<pcl::PointXYZL>::Ptr input(new pcl::PointCloud<pcl::PointXYZL>);
      for(auto it : rg_clusters)
        *input += *it.second;
      const float voxel_leaf=.2;
      pcl::VoxelGrid<pcl::PointXYZL> sor;
      sor.setInputCloud(input);
      sor.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
      sor.filter(*merged_p_cloud);
      merged_p_tree->setInputCloud(merged_p_cloud);
      projectClusters(merged_p_cloud,merged_p_tree, dense_cloud, step, uvs, plane_marker);
    }
    getBgMask(rg_clusters,p_coeffs, merged_p_tree, plane_marker,linear_acc_->cast<float>(), bg_mask);

    cv_bridge::CvImage cv_image;
    {
      cv_image.image = plane_marker;
      cv_image.encoding = sensor_msgs::image_encodings::TYPE_32SC1;
      sensor_msgs::ImagePtr msg = cv_image.toImageMsg();
      res.p_marker = *msg;
    }
    {
      cv_image.image = bg_mask;
      cv_image.encoding = sensor_msgs::image_encodings::TYPE_32SC1;
      sensor_msgs::ImagePtr msg = cv_image.toImageMsg();
      res.p_mask = *msg;
    }
    if(pub_planeseg_.getNumSubscribers() > 0){
      cv::Mat dst = GetColoredLabel(plane_marker);
      HighlightBoundary(plane_marker,dst);
      cv_bridge::CvImage cv_img;
      cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      cv_img.image    = dst;
      pub_planeseg_.publish(*cv_img.toImageMsg());
    }
    if(pub_bgmask_.getNumSubscribers() > 0){
      cv::Mat dst = GetColoredLabel(bg_mask);
      cv_bridge::CvImage cv_img;
      cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      cv_img.image    = dst;
      pub_bgmask_.publish(*cv_img.toImageMsg());
    }
    return true;
  }

  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg){
    if(frame_id_.empty())
      return;
    // TODO linear_acc_
    tf2_ros::Buffer tfbuffer;
    tf2_ros::TransformListener tfListener(tfbuffer);

    // Wait for the required publication to be available
    ros::Duration duration(-1);
    ros::Time now = ros::Time(0);
    int n_try = 0;
    while(!tfbuffer.canTransform(frame_id_, imu_msg->header.frame_id, now)) {
      ros::Duration(0.1).sleep();
      if(n_try++ > 10)
        ROS_INFO("Waiting for transform from %s to %s", imu_msg->header.frame_id.c_str(), frame_id_.c_str());
      now = ros::Time::now();
    }
    geometry_msgs::TransformStamped transform;
    try{
      transform = tfbuffer.lookupTransform(frame_id_, imu_msg->header.frame_id, now);
    }
    catch (tf2::TransformException &ex) {
      ROS_ERROR("%s",ex.what());
      exit(1);
    }
    Eigen::Quaterniond quat;
    //ROS_INFO_STREAM("quat = " << transform.transform.rotation);
    quat.x() = transform.transform.rotation.x;
    quat.y() = transform.transform.rotation.y;
    quat.z() = transform.transform.rotation.z;
    quat.w() = transform.transform.rotation.w;
    Eigen::Vector3d acc0(imu_msg->linear_acceleration.x,
                         imu_msg->linear_acceleration.y,
                         imu_msg->linear_acceleration.z);
    *linear_acc_ = quat * acc0;
    //ROS_INFO_STREAM(linear_acc_->transpose());
    return;
  }

private:
  void rgbCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    rgb_ = cv_ptr->image;
    if(msg->header.frame_id.empty()){
      ROS_ERROR_STREAM("No frame id in rgb");
      exit(1);
    }
    frame_id_ = msg->header.frame_id;
  }

  void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    depth_ = cv_ptr->image;
  }

  void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    // Store latest camera info data in member variable
    const sensor_msgs::CameraInfo info = *msg;
    cv::Mat K, D;
    cvtCameraInfo2CvMat(info, K, D);
    bool rectified = true;
    for(int r=0; r < D.rows; r++){
      if(std::abs(D.at<double>(r,0)) < 1e-5)
        continue;
      rectified=false;
      break;
    }

    nu_map_ = cv::Mat::zeros(info.height, info.width, CV_32FC1);
    nv_map_ = cv::Mat::zeros(info.height, info.width, CV_32FC1);
    if(!rectified){
      // u,v를 모두 모아서, undistortPoints 수행.
      ROS_INFO_STREAM("Distortion model = " << info.distortion_model);
      ROS_ERROR("TODO Not support unrectified image yet)");
      exit(1);
      return;
    }
    else{
      const float& fx = K.at<double>(0,0);
      const float& fy = K.at<double>(1,1);
      const float& cx = K.at<double>(0,2);
      const float& cy = K.at<double>(1,2);
      for(int r=0; r<nu_map_.rows; r++){
        for(int c=0; c<nu_map_.cols; c++){
          nu_map_.at<float>(r,c) = ((float)c - cx)/fx;
          nv_map_.at<float>(r,c) = ((float)r - cy)/fy;
        }
      }
    }
    K_ = K;
    D_ = D;
    return;
  }

  ros::NodeHandle nh_;
  ros::Subscriber rgb_sub_, depth_sub_, camera_info_sub_;
  ros::Publisher pub_points_, pub_rg_, pub_imu_, pub_planeseg_, pub_bgmask_;

  Eigen::Vector3d* linear_acc_;
  std::string frame_id_;
  cv::Mat rgb_, depth_;
  cv::Mat nu_map_, nv_map_;
  cv::Mat K_, D_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "~");
  ros::NodeHandle nh("~");
  bool is_service;
  if (!nh.getParam("is_service", is_service)) {
    ROS_ERROR("Failed to get parameter 'is_service'");
    return 1;
  }
  BgDetector bg_detector(nh);
  if(is_service){
    ros::ServiceServer s0 = nh.advertiseService("GetBg", &BgDetector::GetBg, &bg_detector);
    ros::spin();
  }
  else{
    ros::Rate rate(2.);

    ros::Subscriber imu_sub_(nh.subscribe<sensor_msgs::Imu>("imu",1, &BgDetector::imuCallback,
                                                            &bg_detector));

    while(!ros::isShuttingDown()){
      // TODO IMU sub here
      rate.sleep();
      ros_unet::GetBg::Response res;
      auto t0 = std::chrono::high_resolution_clock::now();
      if(!bg_detector.Process(res)){
        ros::spinOnce();
        continue;
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      auto etime = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
      ROS_INFO_STREAM("Elapsed time = " << etime.count() );
      ros::spinOnce();
    }
  }
  return 0;
}
