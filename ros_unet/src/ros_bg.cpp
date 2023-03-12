#include "ros_util.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>

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
#include <pcl/kdtree/kdtree_flann.h>

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

pcl::PointCloud<pcl::PointXYZ>::Ptr Unproject(cv::Mat depth, cv::Mat nu_map, cv::Mat nv_map,
                                              cv::Mat init_mask,
                                              float voxel_leaf
                                              ){
  const double max_z = 5.; // TODO
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  int n = cv::countNonZero(init_mask);
  cloud->reserve(n);
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
      pcl::PointXYZ pt(nu*z, nv*z, z);
      cloud->points.push_back(pt);
    }
  }
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize (voxel_leaf, voxel_leaf, voxel_leaf);
  sor.filter(*cloud);
  return cloud;
}

std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr > regionGrowing(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
  pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setKSearch(50);
  normal_estimator.compute(*normals);

  pcl::IndicesPtr indices(new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0.0, 1.0);
  pass.filter(*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize(50);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud);
  //reg.setIndices(indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(1.0);

  std::vector<pcl::PointIndices> _clusters;
  reg.extract(_clusters);

  std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr > clusters;
  for(int i =0; i<_clusters.size();i++){
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices(_clusters.at(i)));
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    clusters[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    extract.filter(*clusters[i]);
  }
  return clusters;
}

std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr > planeCluster(const std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr >& clusters) {
  // 1. Detect planes for each region.
  pcl::PointCloud<pcl::PointXYZL>::Ptr normals(new pcl::PointCloud<pcl::PointXYZL>);
  normals->reserve(clusters.size());
  std::map<int, pcl::PointXYZL> l2d;

  for(auto it : clusters){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = it.second;

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Optional
    seg.setOptimizeCoefficients(true);

    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    pcl::PointXYZL normal;
    normal.x = coefficients->values[0];
    normal.y = coefficients->values[1];
    normal.z = coefficients->values[2];
    pcl::PointXYZL d;
    d.x = coefficients->values[3];
    d.y = 0.;
    d.z = 0.;
    normal.label = d.label = it.first;
    normals->push_back(normal);
    l2d[d.label] = d;
    // coefficients->values[0], coefficients->values[1], coefficients->values[2], and coefficients->values[3]
    // represent the plane coefficients a, b, c, and d respectively
  }

  // 2. Euclidean cluster for 'plane coefficient'
  std::vector<std::vector<int> > plane_cluster; {
    pcl::search::KdTree<pcl::PointXYZL>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZL>);
    normal_tree->setInputCloud(normals);
    std::vector<pcl::PointIndices> normal_cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZL> normal_ec;
    const float normal_square_torlerance = 0.1;
    normal_ec.setClusterTolerance(normal_square_torlerance);
    normal_ec.setMinClusterSize(1);
    normal_ec.setSearchMethod(normal_tree);
    normal_ec.setInputCloud(normals);
    normal_ec.extract(normal_cluster_indices);
    for(const pcl::PointIndices& normal_indices : normal_cluster_indices){
      //const auto& normal = normals->at(indices.indices
      pcl::PointCloud<pcl::PointXYZL>::Ptr d_cloud(new pcl::PointCloud<pcl::PointXYZL>);
      d_cloud->reserve(normal_indices.indices.size());
      for(const int& i : normal_indices.indices){
        const auto& normal_l = normals->at(i);
        d_cloud->push_back( l2d.at(normal_l.label) );
      }
      pcl::search::KdTree<pcl::PointXYZL>::Ptr d_tree(new pcl::search::KdTree<pcl::PointXYZL>);
      d_tree->setInputCloud(d_cloud);
      std::vector<pcl::PointIndices> d_cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZL> d_ec;
      const float d_square_torlerance = 0.1;
      d_ec.setClusterTolerance(d_square_torlerance);
      d_ec.setMinClusterSize(1);
      d_ec.setSearchMethod(d_tree);
      d_ec.setInputCloud(d_cloud);
      d_ec.extract(d_cluster_indices);
      for(const pcl::PointIndices& d_indices : d_cluster_indices){
        std::vector<int> g;
        g.reserve(d_indices.indices.size());
        for(const int& j : d_indices.indices){
          const pcl::PointXYZL& d_l = d_cloud->at(j);
          g.push_back(d_l.label);
        }
        plane_cluster.push_back(g);
      }
    }
  }

  // 3. Merge cloud
  std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr > merged_clusters;
  int i = 1;
  for(const std::vector<int>& indices : plane_cluster){
    // create a new point cloud to hold the merged point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // concatenate the point clouds
    for(const int& label : indices){
      pcl::PointCloud<pcl::PointXYZ>::Ptr c0=clusters.at(label);
      *merged_cloud += *c0;
    }
    merged_clusters[i++] = merged_cloud;
  }

  return merged_clusters;
}

class BgDetector {
public:
  BgDetector(ros::NodeHandle nh) :
    nh_(nh),
    rgb_sub_(nh.subscribe<sensor_msgs::Image>("rgb", 1, &BgDetector::rgbCallback, this)),
    depth_sub_(nh.subscribe<sensor_msgs::Image>("depth", 1, &BgDetector::depthCallback, this)),
    camera_info_sub_(nh.subscribe<sensor_msgs::CameraInfo>("info", 1, &BgDetector::cameraInfoCallback, this))
  {
    pub_points_ = nh_.advertise<sensor_msgs::PointCloud2>("points", 1);
    pub_rg_ = nh_.advertise<sensor_msgs::PointCloud2>("rg", 1);
  }

  void Process(){
    if(rgb_.empty())
      return;
    if(depth_.empty())
      return;
    if(nu_map_.empty())
      return;
    cv::Mat init_mask = cv::Mat::ones(depth_.rows,depth_.cols,CV_8UC1);
    const float voxel_leaf=.02;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud 
      = Unproject(depth_, nu_map_, nv_map_, init_mask, voxel_leaf);
    std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr > rg_clusters
      = regionGrowing(cloud);
    std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr > p_clusters 
      = planeCluster(rg_clusters);

    // TODO
    // 4. Get floor, bg candidates
    // 5. Mapping bg plane on 2D image

    if(pub_points_.getNumSubscribers() > 0) {
      pcl::PCLPointCloud2 pcl_pts;
      pcl::toPCLPointCloud2(*cloud, pcl_pts);
      sensor_msgs::PointCloud2 msg;
      pcl_conversions::fromPCL(pcl_pts, msg);
      msg.header.frame_id = "robot";
      pub_points_.publish(msg);
    }
    if(pub_rg_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      //ColorizeSegmentation(rg_clusters,msg);
      ColorizeSegmentation(p_clusters,msg);
      pub_rg_.publish(msg);
    }

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
    if(!rectified){
      ROS_ERROR("TODO Not support unrectified image yet)");
      exit(1);
      return;
    }
    nu_map_ = cv::Mat::zeros(info.height, info.width, CV_32FC1);
    nv_map_ = cv::Mat::zeros(info.height, info.width, CV_32FC1);
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
    return;
  }

  ros::NodeHandle nh_;
  ros::Subscriber rgb_sub_, depth_sub_, camera_info_sub_;
  ros::Publisher pub_points_, pub_rg_;

  cv::Mat rgb_, depth_;
  cv::Mat nu_map_, nv_map_;
  //sensor_msgs::PointCloud2ConstPtr point_cloud_;
  //sensor_msgs::CameraInfoConstPtr camera_info_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "bgdetector");
  ros::NodeHandle nh("~");
  ros::Rate rate(2.);
  BgDetector bg_detector(nh);
  while(!ros::isShuttingDown()){
    auto t0 = std::chrono::high_resolution_clock::now();
    bg_detector.Process();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto etime = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    ROS_INFO_STREAM("Elapsed time = " << etime.count() );
    rate.sleep();
    ros::spinOnce();
  }
  return 0;
}
