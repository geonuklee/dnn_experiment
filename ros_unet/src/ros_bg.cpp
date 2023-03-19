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
               pcl::PointCloud<pcl::PointXYZL>::Ptr& cloud,
               std::vector<cv::Point2i>& uvs
              ){
  const double max_z = 5.; // TODO
  cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>());
  int n = cv::countNonZero(init_mask);
  cloud->reserve(n);
  uvs.reserve(n);
  for(int r=0; r<depth.rows; r++){
    for(int c=0; c<depth.cols; c++){
      float z = depth.at<float>(r,c);
      if(z==0.)
        continue;
      if( init_mask.at<unsigned char>(r,c) <1 )
        continue;
      if(z > max_z)
        continue;
      const float& nu = nu_map.at<float>(r,c);
      const float& nv = nv_map.at<float>(r,c);
      pcl::PointXYZL pt;
      pt.x = nu*z; pt.y = nv*z; pt.z = z; pt.label = -1;
      cloud->points.push_back(pt);
      uvs.push_back(cv::Point2i(c,r));
    }
  }
  return;
}

void regionGrowing(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
                   std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr >& clusters,
                   std::map<int, pcl::PointXYZL>& rg_means
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

  for(int i =0; i<_clusters.size();i++){
    const int label = i+1;
    pcl::PointCloud<pcl::PointXYZL>::Ptr g(new pcl::PointCloud<pcl::PointXYZL>());
    clusters[label] = g;
    const auto& indices = _clusters.at(i).indices;
    g->reserve(indices.size());
    //pt_mean.x = pt_mean.y = pt_mean.z = 0.;
    std::vector<float> vx,vy,vz;
    vx.reserve(indices.size());
    vy.reserve(indices.size());
    vz.reserve(indices.size());
    for(const int& j : indices){
      auto& pt0 = cloud->at(j);
      pt0.label = label;
      g->push_back(pt0);
      vx.push_back(pt0.x);
      vy.push_back(pt0.y);
      vz.push_back(pt0.z);
      //pt_mean.x += pt.x; pt_mean.y += pt.y; pt_mean.z += pt.z;
    }
    std::sort(vx.begin(), vx.end());
    std::sort(vy.begin(), vy.end());
    std::sort(vz.begin(), vz.end());
    pcl::PointXYZL pt_mean;
    pt_mean.label = label;
    pt_mean.x = vx.at(vx.size()/2);
    pt_mean.y = vy.at(vx.size()/2);
    pt_mean.z = vz.at(vx.size()/2);
    rg_means[label] = pt_mean;
  }
  return;
}

void planeCluster(const std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr >& clusters,
                  const std::map<int, pcl::PointXYZL>& rg_means,
                  std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr >& p_clusters,
                  EigenMap<int, Eigen::Vector4f>& p_coeffs
                  ) {
  /*
  가장큰 평면과, 평행한 이웃 평면의 평균 point 사이의 거리로 병합 판정.

   normal_cluster_indices을 cluster 점의 개수 순서로 먼저 정렬,
  가장큰 평면과, 나란한 법선벡터를 가진 neighbor의 mean point를 비교해서 병합.
  */
  // 1. Detect planes for each region.
  pcl::PointCloud<pcl::PointXYZL>::Ptr planes_normal(new pcl::PointCloud<pcl::PointXYZL>);
  planes_normal->reserve(clusters.size());
  std::map<int, float> l2d; // Label to 'd' of plane coeff
  std::vector< std::pair<int, size_t> > index_sizes;
  index_sizes.reserve(clusters.size());

  for(auto it : clusters){
    if(it.first < 1)
      continue;
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = it.second;
    index_sizes.push_back( std::make_pair(index_sizes.size(), cloud->size()) );

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZL> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Optional
    seg.setOptimizeCoefficients(true);

    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.005);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    pcl::PointXYZL normal;
    if(coefficients->values[3] < 0.)
      for(int i=0;i<4;i++)
        coefficients->values[i] *= -1.;
    normal.x = coefficients->values[0];
    normal.y = coefficients->values[1];
    normal.z = coefficients->values[2];
    normal.label = it.first;
    planes_normal->push_back(normal);
    l2d[it.first] = coefficients->values[3];
  }
  // Sort the vector of pairs in descending order of size 
  std::sort(index_sizes.begin(), index_sizes.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second; });
  pcl::search::KdTree<pcl::PointXYZL>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZL>);
  normal_tree->setInputCloud(planes_normal);
  
  const float normal_square_tolerance = .5;
  const float plane2point_tolerance = .15; // Not square
  std::vector<std::vector<int> > plane_cluster;
  std::set<int> closed;
  for(const auto& it : index_sizes){
    const int& i0 = it.first;
    const pcl::PointXYZL& n0 = planes_normal->at(i0);
    const int& l0 = n0.label;
    if(closed.count(l0))
      continue;
    std::vector<int> neighbors;
    std::vector<float> square_distances;
    normal_tree->radiusSearch(n0, normal_square_tolerance, neighbors, square_distances);
    const float& d0 = l2d.at(l0);
    Eigen::Vector4f p0(n0.x,n0.y,n0.z,d0);
    std::vector<int> g = {l0,};
    closed.insert(l0);
    for(const int& i1 : neighbors){
      const pcl::PointXYZL& n1 = planes_normal->at(i1);
      const int& l1 = n1.label;
      if(closed.count(l1))
        continue;
      const pcl::PointXYZL& pt = rg_means.at(l1);
      Eigen::Vector4f p1(pt.x,pt.y,pt.z,1.);
      float err = std::abs( p0.dot(p1) );
      if(err > plane2point_tolerance)
        continue;
      g.push_back(l1);
      closed.insert(l1);
    }
    /* if(g.size() > 1){
      std::string msg = "Merging for ";
      for(const auto&idx : g)
        msg += std::to_string(idx) + ",";
      ROS_INFO_STREAM(msg << p0.transpose() );
    } */
    plane_cluster.push_back(g);
    p_coeffs[l0] = p0;
  }

  // 3. Merge cloud
  for(const std::vector<int>& indices : plane_cluster){
    // create a new point cloud to hold the merged point clouds
    pcl::PointCloud<pcl::PointXYZL>::Ptr merged_plane(new pcl::PointCloud<pcl::PointXYZL>);
    // concatenate the point clouds
    for(const int& label : indices){
      pcl::PointCloud<pcl::PointXYZL>::Ptr c0=clusters.at(label);
      *merged_plane += *c0;
    }
    const int& label = indices.at(0);
    for(auto& pt : *merged_plane)
      pt.label = label;
    p_clusters[label] = merged_plane;
  }
  if(clusters.count(0))
    p_clusters[0] = clusters.at(0);
  return;
}

size_t getNumberOfPointsBehindPlane(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
                                    const Eigen::Vector4f& coeff,
                                    int plane_label,
                                    float distance_th){
  if(distance_th < 0.)
    ROS_WARN_STREAM("distance_th should be positive");
  size_t n = 0;
  for(const auto& pt : *cloud){
    if(pt.label == plane_label)
      continue;
    float d = coeff.dot(Eigen::Vector4f(pt.x,pt.y,pt.z,1.));
    if(d + distance_th < 0.)
      n++;
  }
  return n;
}

void getFloorAndWall(std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr > p_clusters,
                     pcl::PointCloud<pcl::PointXYZL>::Ptr merged_p_cloud,
                     const EigenMap<int, Eigen::Vector4f>& p_coeffs,
                     const cv::Mat marker,
                     const Eigen::Vector3f& floor_norm_prediction,
                     const float distance_th,
                     const float th_floor,
                     int& l_floor,
                     int& l_backwall,
                     cv::Mat& mask
                       ){
  /* 
  * 바닥면 : negative_gravity 와 비슷한 방향의, largest_2d_bottom_seg 를 선택.
  * 뒷면 : norm이 negative z-axis에 가까운 평면 중, 평면-원점이 가장 먼 평면.
  * 필터 조건 : OBB 가로 세로면적이 K[m] 이상 
  * 
  */
  const int max_bottom = 20; // [pixel]  TODO
  const float th_perpen = std::sin(M_PI/180.*80.); // 90deg에 가까울수록 엄밀한,

  l_floor = l_backwall = -1;
  {
    std::map<int, size_t> m_counts;
    for(int r = marker.rows-max_bottom; r< marker.rows; r++){
      for(int c = 0; c < marker.cols; c++){
        const int32_t& l = marker.at<int32_t>(r,c);
        if(l < 1) // Default mask == 0
          continue;
        m_counts[l]++;
      }
    }
    if(!m_counts.empty()){
      std::vector<std::pair<int,size_t> > v_counts;
      v_counts.reserve(m_counts.size());
      for(auto it : m_counts)
        v_counts.push_back(std::make_pair(it.first,it.second));
      // Sort the vector of pairs in descending order of counts 
      std::sort(v_counts.begin(), v_counts.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.second > rhs.second; });
      for(auto it : v_counts){
        const int& l = it.first;
        const Eigen::Vector4f& p = p_coeffs.at(l);
        if(floor_norm_prediction.dot(p.head<3>()) < th_floor)
          continue;
        int n = getNumberOfPointsBehindPlane(merged_p_cloud,p,l,distance_th);
        //ROS_INFO_STREAM("p=" << p.transpose()  << "Out/All=" << n <<"/"<<merged_p_cloud->size() );
        if(n > .1*merged_p_cloud->size() )
          continue;
        l_floor = l;
        std::vector<cv::Point> locations;
        cv::findNonZero(marker == l_floor, locations);
        for (const auto& pt : locations)
          mask.at<int32_t>(pt) = 1;  // FLOOR 1
        break;
      }
    }
  }

  Eigen::Vector3f floor_nvec = floor_norm_prediction;
  if(p_coeffs.count(l_floor))
    floor_nvec = p_coeffs.at(l_floor).head<3>();

  std::vector< std::pair<int,double> > backwalls;
  backwalls.reserve(p_coeffs.size());
  for(auto it : p_coeffs){
    const int& l = it.first;
    if(l==l_floor)
      continue;
    const Eigen::Vector4f p = it.second;
    int n = getNumberOfPointsBehindPlane(merged_p_cloud,p,l,distance_th);
    //ROS_INFO_STREAM("p=" << p.transpose()  << "Out/All=" << n <<"/"<<merged_p_cloud->size() );
    if(n > .1*merged_p_cloud->size())
      continue;
    if( floor_nvec.cross(p.head<3>()).norm() > th_perpen)
      backwalls.push_back(std::make_pair(l,p[3]));
  }
  // Sort the vector of pairs in descending order of distance
  if(!backwalls.empty()){
    std::sort(backwalls.begin(), backwalls.end(), [](const auto& lhs, const auto& rhs) {
              return lhs.second > rhs.second; });
    // TODO 2) backwall이 여러개인 경우는?, 측면뷰는 안따져도 되나?
    l_backwall = backwalls.front().first;
    std::vector<cv::Point> locations;
    cv::findNonZero(marker == l_backwall, locations);
    for (const auto& pt : locations)
      mask.at<int32_t>(pt) = 2; // WALL 2
  }
  for(int r=0; r<marker.rows; r++)
    for(int c=0; c<marker.cols; c++)
      if(marker.at<int32_t>(r,c) < 1)
        mask.at<int32_t>(r,c) = 3; // Non flat
  return;
}

void projectClusters(pcl::PointCloud<pcl::PointXYZL>::Ptr merged_p_cloud,
                     pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
                     const std::vector<cv::Point2i>& uvs,
                     cv::Mat& p_marker,
                     cv::Mat& p_mask
                    ){
  float max_square_tolerance = .05; // [meter] TODO
  max_square_tolerance *= max_square_tolerance;

  pcl::search::Search<pcl::PointXYZL>::Ptr p_tree(new pcl::search::KdTree<pcl::PointXYZL>);
  p_tree->setInputCloud(merged_p_cloud);

  std::vector<int> neighbors(1);
  std::vector<float> square_distance(1);

  for(int i=0; i<uvs.size(); i++){
    const cv::Point2i& uv = uvs.at(i);
    pcl::PointXYZL& pt = cloud->at(i);
    p_tree->nearestKSearch(pt, 1, neighbors, square_distance);
    if(square_distance.empty())
      continue;
    if(square_distance.at(0) > max_square_tolerance)
      continue;
    const pcl::PointXYZL& pt0 = merged_p_cloud->at(neighbors.at(0));
    pt.label = pt0.label; // Labeling cloud
    p_marker.at<int32_t>(uv.y,uv.x) = pt.label; // Project the label
    if(pt.label < 1){ // pcl::PointXYZL.label은 uint, mask는 int
      p_marker.at<int32_t>(uv.y,uv.x) = -1;
      p_mask.at<int32_t>(uv.y,uv.x) = -1;;
    }
  }
  return;
}


void EuclideanFilter(pcl::PointCloud<pcl::PointXYZL>::Ptr sparse_cloud,
                  float square_tolerance,
                  std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr >& dense_clusters,
                  pcl::PointCloud<pcl::PointXYZL>::Ptr& dense_cloud
                  ){

  pcl::PointIndices::Ptr results(new pcl::PointIndices);
  // Creating the KdTree object for the search method of the extraction
  boost::shared_ptr<pcl::search::KdTree<pcl::PointXYZL> > tree(new pcl::search::KdTree<pcl::PointXYZL>());
  tree->setInputCloud(sparse_cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZL> ec;
  ec.setClusterTolerance(square_tolerance);
  ec.setMinClusterSize(100);
  ec.setSearchMethod(tree);
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
    tree->nearestKSearch(dense_pt, 1, neighbors, square_distance);
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


class BgDetector {
public:
  BgDetector(ros::NodeHandle nh) :
    nh_(nh),
    rgb_sub_(nh.subscribe<sensor_msgs::Image>("rgb", 1, &BgDetector::rgbCallback, this)),
    depth_sub_(nh.subscribe<sensor_msgs::Image>("depth", 1, &BgDetector::depthCallback, this)),
    camera_info_sub_(nh.subscribe<sensor_msgs::CameraInfo>("info", 1, &BgDetector::cameraInfoCallback, this)),
    imu_sub_(nh.subscribe<sensor_msgs::Imu>("imu",1, &BgDetector::imuCallback, this)),
    linear_acc_(new Eigen::Vector3d)
  {
    pub_points_ = nh_.advertise<sensor_msgs::PointCloud2>("points", 1);
    pub_rg_ = nh_.advertise<sensor_msgs::PointCloud2>("rg", 1);
    pub_imu_ = nh_.advertise<sensor_msgs::Imu>("output_imu",1);
    pub_dst_mask_ = nh.advertise<sensor_msgs::Image>("dst_mask",1);
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

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
    std::vector<cv::Point2i> uvs;
    Unproject(depth_, nu_map_, nv_map_, init_mask, cloud, uvs);
    pcl::PointCloud<pcl::PointXYZL>::Ptr voxel_cloud; {
      voxel_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      const float voxel_leaf=.02;
      pcl::VoxelGrid<pcl::PointXYZL> sor;
      sor.setInputCloud(cloud);
      sor.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
      sor.filter(*voxel_cloud);
      for(auto& pt : *voxel_cloud)
        pt.label = -1;
    }
    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr > rg_clusters;
    std::map<int, pcl::PointXYZL> rg_means;
    regionGrowing(voxel_cloud, rg_clusters, rg_means);

    // Euclidean cluster in more sparse voxel cloud (for computational cost)
    // Erase outliers from voxel_cloud and rg_clusters
    {
      pcl::PointCloud<pcl::PointXYZL>::Ptr vvoxel_cloud;
      const float vvoxel_leaf=.1;
      pcl::VoxelGrid<pcl::PointXYZL> sor;
      sor.setInputCloud(voxel_cloud);
      vvoxel_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      sor.setLeafSize(vvoxel_leaf, vvoxel_leaf, vvoxel_leaf);
      sor.filter(*vvoxel_cloud);

      const float square_tolerance = .5;
      EuclideanFilter(vvoxel_cloud, square_tolerance, rg_clusters, voxel_cloud);
    }

    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr > p_clusters;
    EigenMap<int, Eigen::Vector4f> p_coeffs;
    planeCluster(rg_clusters,rg_means, p_clusters,p_coeffs);

    // 4. Mapping bg plane on 2D image
    pcl::PointCloud<pcl::PointXYZL>::Ptr merged_p_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    for(auto it : p_clusters)
      *merged_p_cloud += *it.second;

    cv::Mat p_marker = cv::Mat::zeros(depth_.rows,depth_.cols, CV_32SC1);
    cv::Mat p_mask = cv::Mat::zeros(depth_.rows, depth_.cols, CV_32SC1);

    projectClusters(merged_p_cloud, cloud, uvs, p_marker, p_mask);

    // 5. Get floor, bg candidates
    bool given_imu = linear_acc_->norm() > 0.001;
    Eigen::Vector3f floor_norm_prediction
      = given_imu?linear_acc_->normalized().cast<float>():Eigen::Vector3f(0.,-.7,-.3).normalized();
    const float distance_th = given_imu?0.1:.5; // [meter] TODO
    const float th_floor = given_imu?std::cos(M_PI/180.*40.):std::cos(M_PI/180.*70.);

    int l_floor, l_backwall;
    getFloorAndWall(p_clusters,merged_p_cloud,p_coeffs,p_marker,
                    floor_norm_prediction, distance_th, th_floor,
                    l_floor,l_backwall, p_mask);

    if(pub_points_.getNumSubscribers() > 0) {
      pcl::PCLPointCloud2 pcl_pts;
      pcl::toPCLPointCloud2(*cloud, pcl_pts);
      sensor_msgs::PointCloud2 msg;
      pcl_conversions::fromPCL(pcl_pts, msg);
      msg.header.frame_id = frame_id_;
      pub_points_.publish(msg);
    }
    if(pub_rg_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      ColorizeSegmentation(rg_clusters,msg);
      //ColorizeSegmentation(p_clusters,msg);
      msg.header.frame_id = frame_id_;
      pub_rg_.publish(msg);
    }
    {
      cv_bridge::CvImage cv_image;
      cv_image.image = p_marker;
      cv_image.encoding = sensor_msgs::image_encodings::TYPE_32SC1;
      sensor_msgs::ImagePtr msg = cv_image.toImageMsg();
      res.p_marker = *msg;
    }
    {
      cv_bridge::CvImage cv_image;
      cv_image.image = p_mask;
      cv_image.encoding = sensor_msgs::image_encodings::TYPE_32SC1;
      sensor_msgs::ImagePtr msg = cv_image.toImageMsg();
      res.p_mask = *msg;
    }

    // TODO 이거 받아서.. 배경 어떻게 제거?
    for(auto it : p_coeffs) {
      ros_unet::Plane msg;
      msg.id = it.first;
      for(int i=0; i<4; i++)
        msg.coeff[i] = it.second[i];
      res.p_coeffs.push_back(msg);
    }

    if(pub_dst_mask_.getNumSubscribers() > 0){
      cv::Mat dst_p_marker = GetColoredLabel(p_marker);
      HighlightBoundary(p_marker,dst_p_marker);
      cv::Mat dst_bg = GetColoredLabel(p_mask);
      cv::Mat dst;
      cv::hconcat(dst_p_marker,dst_bg,dst);

      cv_bridge::CvImage cv_img;
      cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      cv_img.image    = dst;
      pub_dst_mask_.publish(*cv_img.toImageMsg());
    }
    return true;
  }

private:
  void imuCallback(const sensor_msgs::ImuConstPtr& imu_msg){
    if(frame_id_.empty())
      return;
    //ROS_INFO_STREAM("IMU0 = " << imu_msg->linear_acceleration);
    static tf2_ros::Buffer tf_buffer;
    static tf2_ros::TransformListener tf_listener(tf_buffer);
    try {
      //ROS_INFO_STREAM("frame_id_ = " << imu_msg->header.frame_id << "->" << frame_id_);
      // Get the transform from base_link to map
      geometry_msgs::TransformStamped transform = 
        tf_buffer.lookupTransform(frame_id_, imu_msg->header.frame_id, ros::Time(0));

      Eigen::Quaterniond quat;
      quat.x() = transform.transform.rotation.x;
      quat.y() = transform.transform.rotation.y;
      quat.z() = transform.transform.rotation.z;
      quat.w() = transform.transform.rotation.w;
      Eigen::Vector3d acc0(imu_msg->linear_acceleration.x,
                          imu_msg->linear_acceleration.y,
                          imu_msg->linear_acceleration.z);
      *linear_acc_ = quat * acc0;

      // Create a new IMU message with the transformed orientation
      if(!frame_id_.empty()){
        sensor_msgs::Imu imu_transformed = *imu_msg;
        imu_transformed.header.frame_id = frame_id_;
        imu_transformed.linear_acceleration.x = linear_acc_->x();
        imu_transformed.linear_acceleration.y = linear_acc_->y();
        imu_transformed.linear_acceleration.z = linear_acc_->z();
        pub_imu_.publish(imu_transformed);
      }
      // Publish the transformed IMU message
      // imu_transformed_pub.publish(imu_transformed);
    }
    catch (tf2::TransformException& ex) {
      ROS_INFO("Could not transform IMU data: %s", ex.what());
    }
    return;
  }

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
  ros::Subscriber rgb_sub_, depth_sub_, camera_info_sub_, imu_sub_;
  ros::Publisher pub_points_, pub_rg_, pub_imu_, pub_dst_mask_;

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
    while(!ros::isShuttingDown()){
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
