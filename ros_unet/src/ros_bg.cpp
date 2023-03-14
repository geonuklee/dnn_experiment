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

std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr>
  regionGrowing(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud){
  pcl::search::Search<pcl::PointXYZL>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZL>);
  pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZL, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setKSearch(50);
  normal_estimator.compute(*normals);

  pcl::RegionGrowing<pcl::PointXYZL, pcl::Normal> reg;
  reg.setMinClusterSize(50);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud);
  //reg.setIndices(indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(0.1);

  std::vector<pcl::PointIndices> _clusters;
  reg.extract(_clusters);

  std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr > clusters;
  for(int i =0; i<_clusters.size();i++){
    const int label = i+1;
    pcl::PointCloud<pcl::PointXYZL>::Ptr g(new pcl::PointCloud<pcl::PointXYZL>());
    clusters[label] = g;
    for(const int& j : _clusters.at(i).indices){
      const auto& pt0 = cloud->at(j);
      pcl::PointXYZL pt;
      pt.x = pt0.x; pt.y = pt0.y; pt.z = pt0.z; pt.label = label;
      g->push_back(pt);
    }
  }
  return clusters;
}

void planeCluster(const std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr >& clusters,
                  std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr >& p_clusters,
                  EigenMap<int, Eigen::Vector4f>& p_coeffs
                  ) {
  // 1. Detect planes for each region.
  pcl::PointCloud<pcl::PointXYZL>::Ptr normals(new pcl::PointCloud<pcl::PointXYZL>);
  normals->reserve(clusters.size());
  std::map<int, pcl::PointXYZL> l2d;

  for(auto it : clusters){
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud = it.second;

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZL> seg;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    // Optional
    seg.setOptimizeCoefficients(true);

    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_LMEDS);
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
  std::vector<std::vector<int> > plane_cluster;
  {
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
      Eigen::Vector3d mean_normal(0.,0.,0.);
      for(const int& i : normal_indices.indices){
        const auto& normal_l = normals->at(i);
        d_cloud->push_back( l2d.at(normal_l.label) );
        mean_normal[0] += normal_l.x;
        mean_normal[1] += normal_l.y;
        mean_normal[2] += normal_l.z;
      }
      mean_normal /= (float) normal_indices.indices.size();
      mean_normal /= mean_normal.norm();
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
        float mean_d = 0.;
        for(const int& j : d_indices.indices){
          const pcl::PointXYZL& d_l = d_cloud->at(j);
          g.push_back(d_l.label);
          mean_d += d_l.x;
        }
        mean_d /= d_indices.indices.size();
        plane_cluster.push_back(g);
        Eigen::Vector4f norm(mean_normal.x(),mean_normal.y(),mean_normal.z(),mean_d);
        if(norm[3] < 0.)
          norm *= -1.;
        p_coeffs[plane_cluster.size()] = norm;
      }
    }
  }

  // 3. Merge cloud
  int label = 1;
  for(const std::vector<int>& indices : plane_cluster){
    // create a new point cloud to hold the merged point clouds
    pcl::PointCloud<pcl::PointXYZL>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    // concatenate the point clouds
    for(const int& label : indices){
      pcl::PointCloud<pcl::PointXYZL>::Ptr c0=clusters.at(label);
      *merged_cloud += *c0;
    }
    // TODO relabel
    for(auto& pt : *merged_cloud)
      pt.label = label;
    p_clusters[label] = merged_cloud;
    label++;
  }
  return;
}

cv::Mat getFloorAndWall(const std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr>& p_clusters,
                     const EigenMap<int, Eigen::Vector4f>& p_coeffs,
                     const cv::Mat marker,
                     int& l_floor,
                     int& l_backwall
                     ){
  /* 
  * 바닥면 : negative_gravity 와 비슷한 방향의, largest_2d_bottom_seg 를 선택.
  * 뒷면 : norm이 negative z-axis에 가까운 평면 중, 평면-원점이 가장 먼 평면.
  * 필터 조건 : OBB 가로 세로면적이 K[m] 이상 
  * 
  */
  const int max_bottom = 20; // [pixel]  TODO
  Eigen::Vector3f ideal_floor_nvec(0., -.7, -.3); // TODO Convert it as g_acc from Kinect
  ideal_floor_nvec.normalize();
  const float th_floor = std::cos(M_PI/180.*40.);
  const float th_backwall = std::cos(M_PI/180.*60.);

  l_floor = l_backwall = -1;
  cv::Mat mask = cv::Mat::zeros(marker.rows, marker.cols, CV_32SC1);
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
        if(ideal_floor_nvec.dot(p.head<3>()) < th_floor)
          continue;
        l_floor = l;
        std::vector<cv::Point> locations;
        cv::findNonZero(marker == l_floor, locations);
        for (const auto& pt : locations)
          mask.at<int32_t>(pt) = 1;
        break;
      }
    }
  }
  Eigen::Vector4f ideal_backwall_nvec(0., 0., -1., 1.);
  if(p_coeffs.count(l_floor)){
    const Eigen::Vector3f nvec = p_coeffs.at(l_floor).head<3>();
    ideal_backwall_nvec.head<3>() = Eigen::Vector3f::UnitX().cross(nvec);
  }

  std::vector< std::pair<int,double> > backwalls;
  backwalls.reserve(p_coeffs.size());
  for(auto it : p_coeffs){
    const int& label = it.first;
    const Eigen::Vector4f p = it.second;
    if(ideal_backwall_nvec.dot(p) > th_backwall)
      backwalls.push_back(std::make_pair(label,p[3]));
  }
  // Sort the vector of pairs in descending order of distance
  if(!backwalls.empty()){
    std::sort(backwalls.begin(), backwalls.end(), [](const auto& lhs, const auto& rhs) {
              return lhs.second > rhs.second; });
    l_backwall = backwalls.front().first;
    std::vector<cv::Point> locations;
    cv::findNonZero(marker == l_backwall, locations);
    for (const auto& pt : locations)
      mask.at<int32_t>(pt) = 2;
  }
  return mask;
}

void projectClusters(const std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr>& p_clusters,
                     pcl::PointCloud<pcl::PointXYZL>::Ptr cloud,
                     const std::vector<cv::Point2i>& uvs,
                     cv::Mat& marker
                    ){
  float max_square_tolerance = .05; // [meter] TODO
  max_square_tolerance *= max_square_tolerance;

  pcl::PointCloud<pcl::PointXYZL>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZL>);
  for(auto it : p_clusters)
    *merged_cloud += *it.second;
  pcl::search::Search<pcl::PointXYZL>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZL>);
  tree->setInputCloud(merged_cloud);

  std::vector<int> neighbors(1);
  std::vector<float> square_distance(1);

  for(int i=0; i<uvs.size(); i++){
    const cv::Point2i& uv = uvs.at(i);
    pcl::PointXYZL& pt = cloud->at(i);
    tree->nearestKSearch(pt, 1, neighbors, square_distance);
    if(square_distance.empty())
      continue;
    if(square_distance.at(0) > max_square_tolerance)
      continue;
    const pcl::PointXYZL& pt0 = merged_cloud->at(neighbors.at(0));
    pt.label = pt0.label; // Labeling cloud
    marker.at<int32_t>(uv.y,uv.x) = pt.label; // Project the label
  }
  return;
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

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
    std::vector<cv::Point2i> uvs;
    Unproject(depth_, nu_map_, nv_map_, init_mask, cloud, uvs);
    pcl::PointCloud<pcl::PointXYZL>::Ptr voxel_cloud; {
      voxel_cloud = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      const float voxel_leaf=.02;
      pcl::VoxelGrid<pcl::PointXYZL> sor;
      sor.setInputCloud(cloud);
      sor.setLeafSize (voxel_leaf, voxel_leaf, voxel_leaf);
      sor.filter(*voxel_cloud);
    }
    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr > rg_clusters = regionGrowing(voxel_cloud);
    std::map<int, pcl::PointCloud<pcl::PointXYZL>::Ptr > p_clusters;
    EigenMap<int, Eigen::Vector4f> p_coeffs;
    planeCluster(rg_clusters,p_clusters,p_coeffs);

    // 4. Mapping bg plane on 2D image
    cv::Mat marker = cv::Mat::zeros(depth_.rows,depth_.cols, CV_32SC1);
    projectClusters(p_clusters, cloud, uvs, marker);

    // 5. Get floor, bg candidates
    int l_floor, l_backwall;
    cv::Mat bg_mask = getFloorAndWall(p_clusters,p_coeffs,marker, l_floor,l_backwall);

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
    if(true){
      cv::Mat dst_marker = GetColoredLabel(marker);
      cv::Mat dst_bg = GetColoredLabel(bg_mask);
      cv::Mat dst;
      cv::vconcat(dst_marker,dst_bg,dst);
      cv::imshow("dst", dst);
      char c = cv::waitKey(1);
      if(c=='q')
        exit(1);
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
    K_ = K;
    D_ = D;
    return;
  }

  ros::NodeHandle nh_;
  ros::Subscriber rgb_sub_, depth_sub_, camera_info_sub_;
  ros::Publisher pub_points_, pub_rg_;

  cv::Mat rgb_, depth_;
  cv::Mat nu_map_, nv_map_;
  cv::Mat K_, D_;
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
    //ROS_INFO_STREAM("Elapsed time = " << etime.count() );
    rate.sleep();
    ros::spinOnce();
  }
  return 0;
}
