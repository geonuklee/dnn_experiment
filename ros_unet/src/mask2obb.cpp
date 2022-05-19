#include "mask2obb.h"
#include "ros_util.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

void GetCvMat(const sensor_msgs::CameraInfo& camera_info, cv::Mat& K, cv::Mat& D ){
  K = cv::Mat::zeros(3,3,CV_32F);
  for(int i = 0; i<K.rows; i++)
    for(int j = 0; j < K.cols; j++)
      K.at<float>(i,j) = camera_info.K.data()[3*i+j];
  D = cv::Mat::zeros(camera_info.D.size(),1,CV_32F);
  for (int j = 0; j < D.rows; j++)
    D.at<float>(j,0) = camera_info.D.at(j);
  return;
}

ObbEstimator::ObbEstimator(const MarkerCamera& marker_camera){
  //cv::Mat K, D;
  //GetCvMat(camera_info, K, D);
  cv::Mat K = marker_camera.K_;
  cv::Mat D = marker_camera.D_;

  nu_ = cv::Mat::zeros(marker_camera.image_size_.height, marker_camera.image_size_.width,CV_32F);
  nv_ = cv::Mat::zeros(marker_camera.image_size_.height, marker_camera.image_size_.width,CV_32F);
  for(int r = 0; r < nu_.rows; r++){
    for(int c = 0; c < nu_.cols; c++){
      cv::Point2f uv;
      uv.x = c;
      uv.y = r;
      std::vector<cv::Point2f> vec_uv = {uv};
      pcl::PointXYZ xyz;
      std::vector<cv::Point2f> normalized_pt;
      cv::undistortPoints(vec_uv, normalized_pt, K, D);
      nu_.at<float>(r,c) = normalized_pt[0].x;
      nv_.at<float>(r,c) = normalized_pt[0].y;
    }
  }
}


cv::Point2f ObbEstimator::GetUV(int r, int c) const {
  cv::Point2f nuv;
  nuv.x = nu_.at<float>(r,c);
  nuv.y = nv_.at<float>(r,c);
  return nuv;
}

void ColorizeSegmentation(const std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& clouds,
                          sensor_msgs::PointCloud2& msg){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorized_pointscloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  //colors.at()
  int n = 0;
  for(auto it : clouds)
    n += it.second->points.size();
  colorized_pointscloud->points.reserve(n);

  for(auto it : clouds){
    const int& idx = it.first;
    const auto& color = colors.at(idx % colors.size());
    for(const pcl::PointXYZLNormal& xyz : *it.second){
      pcl::PointXYZRGB pt;
      pt.x = xyz.x;
      pt.y = xyz.y;
      pt.z = xyz.z;
      pt.b = color[0];
      pt.g = color[1];
      pt.r = color[2];
      pt.a = 1.;
      colorized_pointscloud->points.push_back(pt);
    }
  }
  pcl::toROSMsg(*colorized_pointscloud, msg);
  msg.header.frame_id = "robot";
  return;
}

pcl::PointIndices::Ptr EuclideanFilterXYZ(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr given_cloud,
                  const ObbParam& param,
                  bool reserve_best_only
                  ){
  pcl::PointIndices::Ptr results(new pcl::PointIndices);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZLNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZLNormal>);
  tree->setInputCloud (given_cloud);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZLNormal> ec;
  ec.setClusterTolerance(param.euclidean_filter_tolerance);
  ec.setMinClusterSize(0);
  ec.setMaxClusterSize(given_cloud->size());
  ec.setSearchMethod(tree);
  ec.setInputCloud(given_cloud);
  ec.extract(cluster_indices);
  if(cluster_indices.empty())
    return results;

  std::sort(std::begin(cluster_indices),
            std::end(cluster_indices),
            [](const pcl::PointIndices& a, const pcl::PointIndices& b) {
            return a.indices.size() > b.indices.size(); });

  int n = 0;
  for(const pcl::PointIndices& indices : cluster_indices){
    if(indices.indices.size() < param.min_points_of_cluster)
      break;
    n += indices.indices.size();
    if(reserve_best_only)
      break;
  }

  results->indices.reserve(n);

  for(const pcl::PointIndices& indices : cluster_indices){
    if(indices.indices.size() < param.min_points_of_cluster)
      break;
    for(const int& index : indices.indices)
      results->indices.push_back(index);
    if(reserve_best_only)
      break;
  }
  return results;
}



void ObbEstimator::GetSegmentedCloud( const g2o::SE3Quat& Tcw,
                                      cv::Mat rgb,
                                      cv::Mat depth,
                                      cv::Mat mask,
                                      cv::Mat convex_edge,
                                      const ObbParam& param,
                                      std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& clouds,
                                      std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& boundary_clouds,
                                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb
                                    ){
  const float max_depth = 5.; // TODO Paramterize

  // Unproject xyzrgb, segmented_clouds, boundary_clouds

  // Denote that each instance in mask must be detached.
  g2o::SE3Quat Twc = Tcw.inverse();
  cv::Mat dist_transform; {
    const int margin = 5;
    cv::Mat fg = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
    for(int r =0; r<fg.rows;r++){
      for(int c = 0; c<fg.cols;c++){
        if(r < margin || r > mask.rows-margin || c < margin || c > mask.cols-margin)
          continue;
        if(mask.at<int>(r,c)>0
            && depth.at<float>(r,c) > 0.001
            && convex_edge.at<unsigned char>(r,c)==0)
          fg.at<unsigned char>(r,c) = 255;
      }
    }
    // cv::imshow("convex", 255*convex_edge);
    cv::distanceTransform(fg, dist_transform,cv::DIST_L2, cv::DIST_MASK_3);
  }

  // Lamda function to get Gradient of 'distance_transform'
  // The gradient vector indicates inside direction.
  auto GetGx = [&dist_transform](int r, int c){
    float x0 = dist_transform.at<float>(r, c-1);
    float x1 = dist_transform.at<float>(r, c+1);
    return (x1-x0)/2.;
  };
  auto GetGy = [&dist_transform](int r, int c){
    float y0 = dist_transform.at<float>(r-1, c);
    float y1 = dist_transform.at<float>(r+1, c);
    return (y1-y0)/2.;
  };

  auto GetXYZNormal = [&Twc,&depth, &GetGx, &GetGy, this](int r, int c,
                                                          pcl::PointXYZLNormal& xyznormal){
    float gx = GetGx(r,c);
    float gy = GetGy(r,c);
    int du[2], dv[2];
    if(std::abs(gx) > std::abs(gy) ){
      dv[0] = 2;
      dv[1] = -dv[0];
      du[0] = du[1] = (gx > 0 ? 2 : -2);
    }
    else{
      du[0] = 2;
      du[1] = -du[0];
      dv[0] = dv[1] = (gy > 0 ? 2 : -2);
    }
    Eigen::Vector3d X;{
      cv::Point2f nuv = this->GetUV(r,c);
      float z = depth.at<float>(r,c);
      if(z < 0.000001)
        return false;
      X = Eigen::Vector3d(nuv.x*z, nuv.y*z,z);
      Eigen::Vector3d Xw = Twc*X;
      xyznormal.x = Xw[0];
      xyznormal.y = Xw[1];
      xyznormal.z = Xw[2];
    }
    Eigen::Vector3d dX1;{
      int rdr = r+dv[0];
      int cdc = c+du[0];
      cv::Point2f nuv = this->GetUV(rdr,cdc);
      float z = depth.at<float>(rdr,cdc);
      if(z < 0.000001)
        return false;
      if(std::abs(X[2]-z) > 0.05)
        return false;
      dX1 = Eigen::Vector3d(nuv.x*z, nuv.y*z,z) - X;
    }
    Eigen::Vector3d dX2;{
      int rdr = r+dv[1];
      int cdc = c+du[1];
      cv::Point2f nuv = this->GetUV(rdr,cdc);
      float z = depth.at<float>(rdr,cdc);
      if(z < 0.000001)
        return false;
      if(std::abs(X[2]-z) > 0.05)
        return false;
      dX2 = Eigen::Vector3d(nuv.x*z, nuv.y*z,z) - X;
    }
    Eigen::Vector3d n = dX1.cross(dX2);
    n /= n.norm();
    if(n[2] > 0.) // normal computed from cam coordinate. z dir can't be positive.
      n = -n;
    Eigen::Vector3d nw = Twc.rotation()*n;
    xyznormal.normal_x = nw[0];
    xyznormal.normal_y = nw[1];
    xyznormal.normal_z = nw[2];
    return true;
  };


  // Unproject cloud, boundary
  const int boundary = 5;
  const float f_boundary = boundary;
  int max_idx = 0;
  for(int r = boundary; r < depth.rows-boundary; r++){
    for(int c = boundary; c < depth.cols-boundary; c++){
      float z0 = depth.at<float>(r,c);
      if(z0 < 0.000001 || z0 > max_depth)
        continue;
      const int& idx = mask.at<int>(r,c);
      if(idx == 0)
        continue;
      // Noise remove soluiton for stc_k4a_2021-11-18-13-35-58.bag
      // 이것 없으면, 비스듬하게 대각선 뒷면 다른상자에 cloud를 가져와, 상자가 크게 잡히는 버그가 생김.
      const float& pixel_distance = dist_transform.at<float>(r,c);
      cv::Point2f nuv = GetUV(r,c);
      pcl::PointXYZLNormal xyznormal;
      if(! GetXYZNormal(r,c, xyznormal) )
        continue;

      if(pixel_distance > f_boundary){
        if(!clouds.count(idx)){
          pcl::PointCloud<pcl::PointXYZLNormal>::Ptr ptr_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>());
          ptr_cloud->reserve(1000);
          clouds[idx] = ptr_cloud;
        }
        max_idx = std::max(max_idx, idx);
        pcl::PointCloud<pcl::PointXYZLNormal>::Ptr ptr = clouds.at(idx);
        ptr->push_back(xyznormal);
      }
      else {
        cv::Point2i inner_uv, far_inner_uv, outer_uv;{
          float gx = GetGx(r,c);
          float gy = GetGy(r,c);
          float pixel_offset = 10.; // TODO compute from physical length.
          inner_uv.x = (float) c + pixel_offset * gx;
          inner_uv.y = (float) r + pixel_offset * gy;
          far_inner_uv.x = (float) c + 2. * pixel_offset * gx;
          far_inner_uv.y = (float) r + 2. * pixel_offset * gy;
          outer_uv.x = (float) c - pixel_offset * gx;
          outer_uv.y = (float) r - pixel_offset * gy;
        }

        if(mask.at<int>( inner_uv.y, inner_uv.x ) == idx){
          float inner_z = depth.at<float>(inner_uv.y,inner_uv.x);
          float far_inner_z = depth.at<float>(far_inner_uv.y,far_inner_uv.x);
          float outer_z = depth.at<float>(outer_uv.y,outer_uv.x);
          if(outer_z < 0.0001)
            outer_z = 9999.;

          float da = (z0-inner_z);
          float db = (inner_z-far_inner_z);
          bool continue_from_inside = true; //std::abs(da-db) < 0.05;
          bool no_occlusion_from_outside = true;//z0 < outer_z + 0.01;
          if(continue_from_inside && no_occlusion_from_outside){
            if(!boundary_clouds.count(idx)){
              pcl::PointCloud<pcl::PointXYZLNormal>::Ptr ptr_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>());
              ptr_cloud->reserve(1000);
              boundary_clouds[idx] = ptr_cloud;
            }
            pcl::PointCloud<pcl::PointXYZLNormal>::Ptr ptr = boundary_clouds.at(idx);
            ptr->push_back(xyznormal);
          }
        }
      } // // else of pixel_distance > 5.
    } // for c in cols
  } // for r in rows


  std::set<int> erase_list;
  for(auto it_cloud : clouds){
    pcl::VoxelGrid<pcl::PointXYZLNormal> sor;
    sor.setInputCloud(it_cloud.second);
    sor.setLeafSize (param.voxel_leaf, param.voxel_leaf, param.voxel_leaf);
    sor.filter(*it_cloud.second);

    pcl::PointIndices::Ptr indices = EuclideanFilterXYZ(it_cloud.second, param, true);
    if(indices->indices.empty()){
      erase_list.insert(it_cloud.first);
      continue;
    }
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    pcl::ExtractIndices<pcl::PointXYZLNormal> extract;
    extract.setInputCloud(it_cloud.second);
    extract.setIndices(indices);
    extract.setNegative(false);
    extract.filter(*filtered_cloud);
    clouds[it_cloud.first] = filtered_cloud;
  }

  float max_sqr_distance = 2.*param.voxel_leaf;
  max_sqr_distance *= max_sqr_distance;

  for(auto it_boundary : boundary_clouds){
    if(erase_list.count(it_boundary.first) )
      continue;
    if(!clouds.count(it_boundary.first)){
      erase_list.insert(it_boundary.first);
      continue;
    }
    pcl::VoxelGrid<pcl::PointXYZLNormal> sor;
    sor.setInputCloud(it_boundary.second);
    sor.setLeafSize (param.voxel_leaf, param.voxel_leaf, param.voxel_leaf);
    sor.filter(*it_boundary.second);

    pcl::search::KdTree<pcl::PointXYZLNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZLNormal>);
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud = clouds.at(it_boundary.first);
    tree->setInputCloud(cloud);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);
    indices->indices.reserve(it_boundary.second->size());
    for(int i=0; i < it_boundary.second->size(); i++){
      const auto& pt = it_boundary.second->at(i);
      std::vector<int> kindices;
      std::vector<float> kdists;
      tree->nearestKSearch(pt, 1, kindices, kdists);
      if(kindices.empty())
        continue;
      if(kdists.at(0) < max_sqr_distance)
        indices->indices.push_back(i);
    }

    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>);
    pcl::ExtractIndices<pcl::PointXYZLNormal> extract;
    extract.setInputCloud(it_boundary.second);
    extract.setIndices(indices);
    extract.setNegative(false);
    extract.filter(*filtered_cloud);
    if(filtered_cloud->empty()){
      erase_list.insert(it_boundary.first);
      continue;
    }
    boundary_clouds[it_boundary.first] = filtered_cloud;
  }

  // Remove instance without enough number of points at cluster output.
  for(int idx : erase_list){
    if(clouds.count(idx))
      clouds.erase(idx);
    if(boundary_clouds.count(idx))
      boundary_clouds.erase(idx);
  }

  if(xyzrgb.get()){
    for(int r = 0; r < depth.rows; r++){
      for(int c = 0; c < depth.cols; c++){
        float z0 = depth.at<float>(r,c);
        cv::Point2f uv = GetUV(r,c);
        Eigen::Vector3d Xc(uv.x * z0, uv.y*z0, z0);
        Eigen::Vector3d Xw = Twc * Xc;
        pcl::PointXYZRGB pt;
        pt.x = Xw[0]; pt.y = Xw[1]; pt.z = Xw[2];
        pt.b = rgb.at<cv::Vec3b>(r,c)[0];
        pt.g = rgb.at<cv::Vec3b>(r,c)[1];
        pt.r = rgb.at<cv::Vec3b>(r,c)[2];
        pt.a = 1.;
        xyzrgb->points.push_back(pt);
      }
    }
  }
  return;
}

ObbProcessVisualizer::ObbProcessVisualizer(const std::string& cam_id, ros::NodeHandle& nh)
  : cam_id_(cam_id)
{
  std::string cam_name = cam_id+"/";
  pub_mask     = nh.advertise<sensor_msgs::Image>(cam_name+"mask",2);
  pub_clouds   = nh.advertise<sensor_msgs::PointCloud2>(cam_name+"clouds",2);
  pub_boundary = nh.advertise<sensor_msgs::PointCloud2>(cam_name+"boundary",2);
  pub_contour  = nh.advertise<visualization_msgs::MarkerArray>(cam_name+"contour",2);
  pub_pose0  = nh.advertise<geometry_msgs::PoseArray>(cam_name+"pose0",2);
  pub_pose  = nh.advertise<geometry_msgs::PoseArray>(cam_name+"pose",2);
  pub_unsynced_obb  = nh.advertise<visualization_msgs::MarkerArray>(cam_name+"unsynced_obb",2);
}

sensor_msgs::PointCloud2 ObbProcessVisualizer::Convert(const std::map<int,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_pc(new pcl::PointCloud<pcl::PointXYZRGB>); {
    int n = 0;
    for(auto it: clouds)
      n += it.second->size();
    vis_pc->reserve(n);
  }
  for(auto it: clouds){
    const auto& color = colors.at(it.first%colors.size());
    for(const auto& pt : *it.second){
      pcl::PointXYZRGB xyzrgb;
      xyzrgb.x = pt.x; xyzrgb.y = pt.y; xyzrgb.z = pt.z;
      xyzrgb.r = color[2]; xyzrgb.g = color[1]; xyzrgb.b = color[0];
      vis_pc->push_back(xyzrgb);
    }
  }
  sensor_msgs::PointCloud2 topic;
  pcl::toROSMsg(*vis_pc, topic);
  topic.header.frame_id = "robot";
  return topic;
}

void ObbProcessVisualizer::VisualizeBoundaryClouds(
  const std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) {
  auto topic = Convert(clouds);
  pub_boundary.publish(topic);
}
void ObbProcessVisualizer::VisualizeClouds(
  const std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) {
  auto topic = Convert(clouds);
  pub_clouds.publish(topic);
}

void ObbProcessVisualizer::PutContour(
  const visualization_msgs::Marker& contour_marker){
  contours_current_.markers.push_back(contour_marker);
}

void ObbProcessVisualizer::PutPose0(const geometry_msgs::Pose& pose0) {
  pose0_array_.poses.push_back(pose0);
}

void ObbProcessVisualizer::PutPose(const geometry_msgs::Pose& pose) {
  pose_array_.poses.push_back(pose);
}

void ObbProcessVisualizer::PutUnsyncedOBB(const visualization_msgs::Marker& unsynced_obb) {
  assert(unsynced_obb.type == visualization_msgs::Marker::CUBE);
  obb_current_.markers.push_back(unsynced_obb);
}

ObbProcessVisualizer::~ObbProcessVisualizer() {
}

void ObbProcessVisualizer::Visualize() {

  {
    visualization_msgs::Marker a;
    a.action = visualization_msgs::Marker::DELETEALL;

    visualization_msgs::MarkerArray topic = contours_current_;
    topic.markers.push_back(a);
    std::reverse(topic.markers.begin(), topic.markers.end());

    pub_contour.publish(topic);
    contours_old_ = contours_current_;
    contours_current_.markers.clear();
  }

  pose0_array_.header.frame_id = "robot";
  pub_pose0.publish(pose0_array_);
  pose0_array_.poses.clear();

  pose_array_.header.frame_id = "robot";
  pub_pose.publish(pose_array_);
  pose_array_.poses.clear();

  return;
}

visualization_msgs::MarkerArray ObbProcessVisualizer::GetUnsyncedOBB() {
  visualization_msgs::Marker a;
  a.action = visualization_msgs::Marker::DELETEALL;

  visualization_msgs::MarkerArray output = obb_current_;
  output.markers.push_back(a);
  std::reverse(output.markers.begin(), output.markers.end());
  obb_current_.markers.clear();

  pub_unsynced_obb.publish(output);
  return output;
}


Eigen::Vector3d Cvt2EigenXYZ(const pcl::PointXYZLNormal& pt){
  return Eigen::Vector3d(pt.x, pt.y, pt.z);
}

double GetError(const g2o::SE3Quat& T1w,
                const Eigen::Vector3d& xw,
                const Eigen::Vector3d& min_x1,
                const Eigen::Vector3d& max_x1
                ){
  const Eigen::Vector3d x1 = T1w*xw;
  double err_x = std::min( std::abs(x1[0]-min_x1[0]), std::abs(x1[0]-max_x1[0]) );
  double err_y = std::min( std::abs(x1[1]-min_x1[1]), std::abs(x1[1]-max_x1[1]) );
  double err = std::min(err_x, err_y);
  if(err < 0.01)
    err = 0.;
  return err;
}

int GetBoxInlier(const g2o::SE3Quat& Twl, const Eigen::Vector3d& whd,
                 pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud
                 ){
  const g2o::SE3Quat Tlw = Twl.inverse();
  const Eigen::Vector3d cp = Twl.translation();
  Eigen::Matrix3d Rlw = Tlw.rotation().matrix();
  EigenVector<Eigen::Vector4d> planes;
  planes.reserve(6);
  for(int i=0; i<3; i++){
    for(int k = 0; k <2; k++){
      Eigen::Vector3d nvec = Rlw.row(i);
      if(k > 0)
        nvec = -nvec;
      Eigen::Vector3d x = cp + 0.5 * whd[i] * nvec;
      Eigen::Vector4d plane;
      plane.head<3>() = nvec;
      plane[3] = - nvec.dot(x);
      planes.push_back(plane);
    }
  }

  int n_inlier = 0;
  for(const pcl::PointXYZLNormal& pt : *cloud){
    Eigen::Vector4d x1(pt.x,pt.y,pt.z,1.);
    Eigen::Vector3d pt_norm(pt.normal_x,pt.normal_y,pt.normal_z);
    // assert(pt_norm.norm() < 1e-5);
    for(size_t i=0; i< planes.size(); i++){
      const auto& plane = planes.at(i);
      // Consider poor depth resolution for skewed plane.
      double max_dist_err = (i==0) ? 0.01 : 0.02;
      double min_cos      = (i==0) ? 0.9 : 0.9;
      double dist_err = std::abs( plane.dot(x1) ); 
      if(dist_err > max_dist_err) // TODO parameterize hard coded param
        continue;
      double cos = pt_norm.dot(plane.head<3>() );
      if(cos < min_cos)
        continue;
      n_inlier++;
      break;
    }
  }
  return n_inlier;
}

std_msgs::ColorRGBA GetColor(int id){
  const auto& color = colors.at(id%colors.size());
  std_msgs::ColorRGBA rgba;
  rgba.a = 0.8;
  rgba.r = color[2]/255.;
  rgba.g = color[1]/255.;
  rgba.b = color[0]/255.;
  return rgba;
}

visualization_msgs::Marker GetMarker(const std::shared_ptr<unloader_msgs::Object> obj){
  visualization_msgs::Marker marker;
  marker.color = GetColor(obj->instance_id);
  marker.header.frame_id = obj->point_cloud.header.frame_id;
  marker.id = obj->instance_id;

  if(obj->type == unloader_msgs::Object::SACK ||
     obj->type == unloader_msgs::Object::POUCH ){
    pcl::PointCloud<pcl::PointXYZ>  cloud;
    pcl::fromROSMsg(obj->point_cloud, cloud);
    marker.type = visualization_msgs::Marker::POINTS;
    marker.points.reserve(cloud.size());
    for(const auto& pt : cloud){
      geometry_msgs::Point p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      marker.points.push_back(p);
    }
    marker.scale.x = marker.scale.y = marker.scale.z = 0.01;
  }
  else{
    marker.type = visualization_msgs::Marker::CUBE;
    marker.pose = obj->center_pose; // center of 3D bounding box.
    marker.scale.x = obj->size.x;
    marker.scale.y = obj->size.y;
    marker.scale.z = obj->size.z;
  }
  return marker;
}

pcl::PointCloud<pcl::PointXYZLNormal>::Ptr FilterEuclideanOnPlane(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud,
                                                           const Eigen::Vector4f& plane,
                                                           const ObbParam& param
                                                           ){
  pcl::PointCloud<pcl::PointXYZLNormal>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>());
  {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    inliers->indices.reserve(cloud->points.size());
    for(size_t i = 0; i < cloud->points.size(); i++){
      const pcl::PointXYZLNormal& pt = cloud->points[i];
      float e = Eigen::Vector4f(pt.x, pt.y, pt.z, 1.).dot(plane);
      if(std::abs(e) > param.max_surface_error_)
        continue;

      // normal vector를 기준으로한 filtering
      float cos = Eigen::Vector3f(pt.normal_x, pt.normal_y, pt.normal_z).dot(plane.head<3>());
      if(cos < 0.95)
        continue;
      inliers->indices.push_back(i);
    }
    if(inliers->indices.size() > 4){
      // Reserve the inliner points of champion as input of RANSAC plane detection.
      pcl::ExtractIndices<pcl::PointXYZLNormal> extract;
      extract.setInputCloud(cloud);
      extract.setIndices(inliers);
      extract.setNegative(false);
      extract.filter(*filtered_cloud);
    }
  }

  // Euclidean filter after plane filter for cloud points.
  // if(!filtered_cloud->empty() ) {
  //   pcl::PointIndices::Ptr inliers = EuclideanFilterXYZ(filtered_cloud, param, true);
  //   if(inliers->indices.size() > 10){
  //     pcl::ExtractIndices<pcl::PointXYZLNormal> extract;
  //     extract.setInputCloud(filtered_cloud);
  //     extract.setIndices(inliers);
  //     extract.setNegative(false);
  //     extract.filter(*filtered_cloud);
  //   }
  // }
  return filtered_cloud;
}


bool ComputeBoxOBB(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud,
                pcl::PointCloud<pcl::PointXYZLNormal>::Ptr boundary,
                const ObbParam& param,
                const Eigen::Vector3f& depth_dir,
                const Eigen::Vector3f& t_wc,
                std::shared_ptr<unloader_msgs::Object> obj,
                std::shared_ptr<ObbProcessVisualizer> visualizer,
                const std::vector<float>& floor_plane
               )
{
    // Compute concavehull for (inner) cloud.
    pcl::ConcaveHull<pcl::PointXYZLNormal> chull;
    chull.setAlpha(param.concavehull_alpha_);
    std::vector< pcl::Vertices > polygons;
    chull.setInputCloud(cloud);
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZLNormal>);
    chull.reconstruct(*cloud_hull, polygons);
    if(cloud_hull->empty() ){
      ROS_INFO("Failed to compute concavehull for instance %d", obj->instance_id);
      return false;
    }

    // Each surface of concavehull are considered as candidate for front plane.
    Eigen::Vector3f u0;
    int best_polygon_idx = -1;
    pcl::PointIndices::Ptr surface(new pcl::PointIndices);
    for(int it = 0; it < polygons.size(); it++){
      const pcl::Vertices& polygon = polygons.at(it);
      size_t n_vertices = polygon.vertices.size();

      const uint32_t& idx0 = polygon.vertices.at(0);
      const uint32_t& idx1 = polygon.vertices.at(1);
      const uint32_t& idx2 = polygon.vertices.at(2);
      const pcl::PointXYZLNormal& pcl_pt0 = cloud_hull->at(idx0);
      const pcl::PointXYZLNormal& pcl_pt1 = cloud_hull->at(idx1);
      const pcl::PointXYZLNormal& pcl_pt2 = cloud_hull->at(idx2);
      Eigen::Vector3f p_candi(pcl_pt0.x, pcl_pt0.y, pcl_pt0.z);
      Eigen::Vector3f u0_candi(pcl_pt1.x-pcl_pt0.x, pcl_pt1.y-pcl_pt0.y, pcl_pt1.z-pcl_pt0.z);
      Eigen::Vector3f v0_candi(pcl_pt2.x-pcl_pt0.x, pcl_pt2.y-pcl_pt0.y, pcl_pt2.z-pcl_pt0.z);
      u0_candi.normalize();

      // If the offset between the normal vector and the depth direction is greater than the threshold,
      // exclude it from the candidate list.
      Eigen::Vector3f n0_candi = u0_candi.cross(v0_candi).normalized();
      double cos_dir = std::abs( n0_candi.dot(depth_dir) );

      if(cos_dir < param.min_cos_dir)
        continue;

      // Count number of the inlier points which are close to plane.
      pcl::PointIndices::Ptr surface_candi(new pcl::PointIndices);
      surface_candi->indices.reserve(cloud->size());
      int i = 0;
      for(const pcl::PointXYZLNormal& pt : *cloud) {
        double e = n0_candi.dot( Eigen::Vector3f(pt.x, pt.y, pt.z)-p_candi );
        if(std::abs(e) < param.max_surface_error_)
          surface_candi->indices.push_back(i);
        i++;
      }

      // The candidate plane with maximum inliner points are chosen as an champion.
      if(surface_candi->indices.size() > surface->indices.size() ){
        u0 = u0_candi;
        surface = surface_candi;
        best_polygon_idx = it;
      }
    }

    if(best_polygon_idx < 0){
      ROS_INFO("Failed to get front plane for yolact instance %d", obj->instance_id);
      return false;
    }

    // Compensate n0 (normal vector 0) with RANSAC
    Eigen::Vector4f plane;
    Eigen::Vector3f v0, n0;
    int n_front_inliers = 0;
    {
      // Reserve the inliner points of champion as input of RANSAC plane detection.
      pcl::ExtractIndices<pcl::PointXYZLNormal> extract;
      pcl::PointCloud<pcl::PointXYZLNormal>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZLNormal>() );
      extract.setInputCloud(cloud);
      extract.setIndices(surface);
      extract.setNegative(false);
      extract.filter(*input_cloud);

      // Do RANSACT for plane detection.
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      pcl::SACSegmentation<pcl::PointXYZLNormal> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      //seg.setMethodType(pcl::SAC_LMEDS);
      seg.setDistanceThreshold (0.01);
      seg.setInputCloud(input_cloud);

      // The coefficients are consist with Hessian normal form : [normal_x normal_y normal_z d].
      // ref : https://pointclouds.org/documentation/group__sample__consensus.html
      seg.segment(*inliers, *coefficients);
      n_front_inliers = inliers->indices.size();
      if(n_front_inliers <param.min_points_of_cluster){
        ROS_INFO("Failed to compute plane for yolact instance %d", obj->instance_id);
        return false;
      }

      for(int i =0; i <4; i++)
        plane[i] = coefficients->values.at(i);

      assert( std::abs( plane.head<3>().norm()-1.) < 1e-5 );

      // If normal vector is inverted, correct it.
      Eigen::Vector4f a(t_wc[0],t_wc[1],t_wc[2],1.);
      if(plane.dot(a) < 0.)
         plane = -plane;

      n0 = plane.head<3>();
    }

    // Exclude boundary points departed from plane.
    *boundary =  *FilterEuclideanOnPlane(boundary, plane, param);
    if(boundary->empty()){
      return false;
    }

    /*
    Calculate SE(3) transformation to {w}orld coordinate from {0} coordinate.
    Denote that
     * Center of {0} coordinate is an 1st vertex of 'champion' polygon on concavehull.
     * x-axis is parrel to the line between 1st and 2nd vertex of champion polygon.
     * z-axis is the 'n0' normal vector computed by RANSAC
     * y-axis is cross product of z and x-axis.
    */
    g2o::SE3Quat Tw0; {
      Eigen::Vector3f cp;
      int n_front_surface = 0;
      cp.setZero();
      const Eigen::Vector3f& nvec = plane.head<3>();
      for(const pcl::PointXYZLNormal& pt : *cloud) {
        Eigen::Vector3f xyz(pt.x,pt.y,pt.z);
        float d = nvec.dot(xyz) + plane[3];
        Eigen::Vector3f proj = xyz - d*nvec;
        cp += proj;
      }
      cp /= (float) cloud->size();
      v0 = n0.cross(u0);
      Eigen::Matrix<double,3,3> R0w;
      R0w.row(0) = Eigen::Vector3d(u0[0], u0[1], u0[2]);
      R0w.row(1) = Eigen::Vector3d(v0[0], v0[1], v0[2]);
      R0w.row(2) = Eigen::Vector3d(n0[0], n0[1], n0[2]);
      Tw0 = g2o::SE3Quat(R0w.transpose(), Eigen::Vector3d(cp[0],cp[1],cp[2]) );
    }

  {
    // The contour of convexhull which is computed from projection of boundary points on front plane.
    // TODO Future works - 단순화된 contour 궤적을 구하면, rotation caliper에서 집중되는 시간낭비를 줄일 수 있다.
    std::vector<cv::Point2f> projected_points, contour;
    projected_points.reserve(boundary->size()+cloud->size() );

    //pcl::PointCloud<pcl::PointXYZLNormal>::Ptr surf_cloud = FilterEuclideanOnPlane(cloud, plane, param);
    //for(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr pc_ptr : {boundary, surf_cloud} ) {
    for(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr pc_ptr : {boundary} ) {
    for(const pcl::PointXYZLNormal& pt : *pc_ptr){
      Eigen::Vector3d x0 = Tw0.inverse() * Cvt2EigenXYZ(pt);
      projected_points.push_back(cv::Point2f(x0[0],x0[1]) );
    }
    }
    cv::convexHull(projected_points, contour);

    // Visualize contours
    {
      visualization_msgs::Marker contour_marker;
      contour_marker.header.frame_id = "robot";
      contour_marker.id = obj->instance_id;
      //contour_marker.color = GetColor(obj->instance_id);
      contour_marker.color = GetColor(1);

      contour_marker.points.reserve(2*contour.size());
      for(size_t i =0; i < contour.size(); i++){
        const cv::Point2f& uv0 = contour.at(i);
        const cv::Point2f& uv1 = contour.at( (i+1)%contour.size() );
        geometry_msgs::Point msg;
        msg.x = uv0.x; msg.y = uv0.y; msg.z = 0.;
        contour_marker.points.push_back(msg);
        msg.x = uv1.x; msg.y = uv1.y; msg.z = 0.;
        contour_marker.points.push_back(msg);
      }
      contour_marker.type = visualization_msgs::Marker::LINE_LIST;
      contour_marker.scale.x = 0.02;
      contour_marker.pose.orientation.w = Tw0.rotation().w();
      contour_marker.pose.orientation.x = Tw0.rotation().x();
      contour_marker.pose.orientation.y = Tw0.rotation().y();
      contour_marker.pose.orientation.z = Tw0.rotation().z();
      contour_marker.pose.position.x = Tw0.translation().x();
      contour_marker.pose.position.y = Tw0.translation().y();
      contour_marker.pose.position.z = Tw0.translation().z();
      visualizer->PutContour(contour_marker);
      visualizer->PutPose0(contour_marker.pose);
    }

    // Rotate caliper along contour to find OBB with minimum 'cost'.
    double optimal_offset_error = 99999999.f;
    double optimal_inlier_ratio = 0.;
    Eigen::Vector3d whd;
    g2o::SE3Quat Twl, T1l;
    bool no_solution = true;
    for(size_t i =0; i < contour.size(); i++){
      const cv::Point2f& uv0 = contour.at(i);
      const cv::Point2f& uv1 = contour.at( (i+1)%contour.size() );

      Eigen::Vector3d u1(uv1.x-uv0.x, uv1.y-uv0.y, 0.);
      u1.normalize();

      // The rotation maps to {1} from {0}
      // The axis of {1} coordinate are aligned to optimal OBB.
      Eigen::Matrix<double,3,3> R10;
      R10.setZero();
      R10(2,2) = 1.;
      R10.row(0) = u1.transpose();
      R10.row(1) = R10.row(2).cross(u1.transpose());
      g2o::SE3Quat T01;
      T01.setRotation( g2o::Quaternion(R10.transpose()) );
      g2o::SE3Quat T1w = (Tw0 * T01).inverse();

      double inf = 999999.;
      Eigen::Vector3d min_x1(inf, inf, inf);
      //Eigen::Vector3d min_x1(inf, inf, inf);
      Eigen::Vector3d max_x1(-inf, -inf, 0.);

      // Use boundary+inner points for defining OBB size instead of cloud (points),
      // I thought cloud (points) have no adventage for size accuracy,
      // But actually it has for rotated boxes.
      //for(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr pc_ptr : {boundary, surf_cloud} ) {
      for(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr pc_ptr : {boundary} ) {
        for(const pcl::PointXYZLNormal& pt : *pc_ptr) {
          Eigen::Vector3d x1 = T1w * Cvt2EigenXYZ(pt);
          // 표면의 points만 사용해서, 강인한 결과값 획득하기 위한 조건문.
          if(x1[2] > - param.max_surface_error_){
            for(size_t k=0; k<2; k++){
              min_x1[k] = std::min<double>(min_x1[k], x1[k]);
              max_x1[k] = std::max<double>(max_x1[k], x1[k]);
            }
          }
        }
      }

      for(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr pc_ptr : {boundary, cloud} ){
        for(const pcl::PointXYZLNormal& pt : *pc_ptr) {
          Eigen::Vector3d x1 = T1w * Cvt2EigenXYZ(pt);
          min_x1[2] = std::min<double>(min_x1[2], x1[2]);
        }
      }


      {
        double w = max_x1[0]-min_x1[0];
        double h = max_x1[1]-min_x1[1];
        double min_depth = 0.2; // 1. * std::min(w,h);
        if(min_x1[2] > -min_depth)
          min_x1[2] = -min_depth;
      }


      // If points on front plane are too less considering size of it,
      //  exclude it from candidates because something is wrong.
      double area = (max_x1[0]-min_x1[0])*(max_x1[1]-min_x1[1]);
      int expected_front_points = area/param.voxel_leaf/param.voxel_leaf;
      float ratio = (float) n_front_inliers / (float) expected_front_points;
      if(ratio < param.min_visible_ratio_of_frontplane)
        continue;

      const Eigen::Vector3d whd_candidate = max_x1-min_x1;
      Eigen::Vector3d cp1 = 0.5*(max_x1+min_x1);

      // Get champoin of rotate caliper with minimum offset error(=cost).
      double offset_error = 0.;
      for(pcl::PointCloud<pcl::PointXYZLNormal>::Ptr pc_ptr : {boundary, cloud} ) {
        for(const pcl::PointXYZLNormal& pt : *pc_ptr) {
          offset_error += GetError(T1w,  Cvt2EigenXYZ(pt), min_x1, max_x1);
        }
      }

      if(offset_error < optimal_offset_error ){
        no_solution=false;
        optimal_offset_error = offset_error;
        whd = whd_candidate;

        // The center point of {1} coordinate is positioned on center of OBB.
        T1l.setTranslation(cp1);
        Twl = T1w.inverse() * T1l;
      }
    }

    if(no_solution){
      ROS_INFO("Failed-too less points for big front plane- for yolact instance %d", obj->instance_id);
      return false;
    }

    const Eigen::Matrix<double,3,3> Rwl0 = Twl.rotation().toRotationMatrix();
    int c0, c1, c2;
    double val_max0 = 0.;
    for(int k = 0; k<3; k++){
      const double& val = Rwl0(0,k);
      if(std::abs(val) > std::abs(val_max0) ){
        c0 = k;
        val_max0 = val;
      }
    }
    double val_max1 = 0.;
    for(int k = 0; k<3; k++){
      if(k==c0)
        continue;
      const double& val = Rwl0(2,k);
      if(std::abs(val) > std::abs(val_max1) ){
        c1 = k;
        val_max1 = val;
      }
    }
    for(int k=0; k <3; k++){
      if(k==c0)
        continue;
      if(k==c1)
        continue;
      c2 = k;
      break;
    }

    Eigen::Matrix<double,3,3> Rwl;
    auto whd0 = whd;
    Rwl.col(0) = Rwl0.col(c0);
    if(val_max0 > 0.)  //  box detection 의 x축 방향 결정.
      Rwl.col(0) = -Rwl.col(0);
    Rwl.col(1) = Rwl0.col(c1);
    if(val_max1 < 0.)
      Rwl.col(1) = -Rwl.col(1);
    whd[0] = whd0[c0];
    whd[1] = whd0[c1];
    whd[2] = whd0[c2];
    Rwl.col(2) = Rwl.col(0).cross(Rwl.col(1));

    // Twl : {l} coordinate is the final local coordinate of OBB,
    // which has aligned axis on OBB and origin is positioned at center of OBB.
    Twl.setRotation(g2o::Quaternion(Rwl));

    int n_inlier = GetBoxInlier(Twl, whd, cloud);
    int N = cloud->size();
    n_inlier += GetBoxInlier(Twl, whd, boundary);
    N += boundary->size();
    float ratio = (float) n_inlier / (float) N;
    //if(ratio < 0.8){
    //  ROS_INFO("Not enough inlier xyzl for instance %d", obj->instance_id);
    //  return false;
    //}

    if(!floor_plane.empty()){
      // Check whether the box is floor or not.
      Eigen::Vector4d plane(floor_plane[0], floor_plane[1], floor_plane[2], floor_plane[3]);
      //std::cout << "floor_plane = " << plane.transpose() << std::endl;
      const auto& t = Twl.translation();
      Eigen::Vector4d cp(t[0], t[1], t[2], 1.);
      if(plane.dot(cp) < 0.)
        return false;
    }

    obj->center_pose.position.x = Twl.translation().x();
    obj->center_pose.position.y = Twl.translation().y();
    obj->center_pose.position.z = Twl.translation().z();
    obj->center_pose.orientation.w = Twl.rotation().w();
    obj->center_pose.orientation.x = Twl.rotation().x();
    obj->center_pose.orientation.y = Twl.rotation().y();
    obj->center_pose.orientation.z = Twl.rotation().z();
    obj->size.x = whd[0];
    obj->size.y = whd[1];
    obj->size.z = whd[2];
    {
      auto marker = GetMarker(obj);
      visualizer->PutUnsyncedOBB(marker);
      visualizer->PutPose(obj->center_pose);
    }

    // Compute visible plane
    {
      g2o::SE3Quat Tlw = Twl.inverse();
      std::vector<int> inlier_for_eachplane;
      inlier_for_eachplane.resize(6);
      for(size_t i = 0; i < cloud->size(); i++) {
        const pcl::PointXYZLNormal& pt = cloud->at(i);
        const Eigen::Vector3d xl = Tlw*Cvt2EigenXYZ(pt);
        std::vector<float> distances, plane_thresholds;
        distances.resize(6);
        plane_thresholds.resize(6);
        for(int k = 0; k <3; k++){
          float plane_threshold = 0.2*whd[k];
          float err_neg = std::abs( xl[k] + whd[k]/2.);
          float err_pos = std::abs( xl[k] - whd[k]/2.);

          plane_thresholds[2*k] = plane_threshold;
          plane_thresholds[2*k+1] = plane_threshold;

          if(err_pos < err_neg){
            distances[2*k] = err_pos;
            distances[2*k+1] = 999999.;
          }
          else{
            distances[2*k+1] = err_neg;
            distances[2*k] = 999999.;
          }
        }

        int nplane = 0;
        for(int k = 0; k < 6; k++){
          if(distances.at(k) < plane_thresholds.at(k))
            nplane++;
        }
        if(nplane<2){
          for(int k = 0; k < 6; k++)
            if(distances.at(k) < plane_thresholds.at(k))
              inlier_for_eachplane[k]++;
        }
      }

      int n_inlier =0;
      for(const int& n : inlier_for_eachplane)
        n_inlier += n;

      int min_points_of_plane = std::max<int>(0.1 * n_inlier,1);
      int nvisible = 0;
      for(size_t k=0; k < 6; k++){
        if(inlier_for_eachplane.at(k) < min_points_of_plane)
          continue;
        obj->visible_plane[k] = true;
        nvisible++;
      }

      if(nvisible<1){
        obj->visible_plane[0] = true;
        ROS_WARN("Failed to compute visible plane"); // 바라보는 정면이 visible이라 가정.
      }
    }
  }

  // Provide 'points cloud' for Pouch object.
  if(obj->type==unloader_msgs::Object::POUCH){
    pcl::PointCloud<pcl::PointXYZLNormal> target_cloud = *cloud + *boundary;
    pcl::toROSMsg(target_cloud, obj->point_cloud);
    obj->point_cloud.header.frame_id = "robot";
  }
  return true;
}




void ObbEstimator::ComputeObbs(const std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& segmented_clouds,
                   const std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& boundary_clouds,
                   const ObbParam& param,
                   const g2o::SE3Quat& Tcw,
                   const std::string& cam_id,
                   std::shared_ptr<ObbProcessVisualizer> visualizer,
                   const std::vector<float>& floor_plane
                   ) {

  Eigen::Vector3f depth_dir, t_wc;
  {
    const Eigen::Vector3d& dir_d = Tcw.rotation().matrix().row(2);
    depth_dir = dir_d.cast<float>();
    t_wc = Tcw.inverse().translation().cast<float>();
  }

  // The loop for each instance.
  for(auto it : segmented_clouds){
    // Pass the instance if there is no boundary
    if(! boundary_clouds.count(it.first)){
      ROS_INFO("No boundary clouds for instance %d", it.first);
      continue;
    }
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr cloud = it.second;
    pcl::PointCloud<pcl::PointXYZLNormal>::Ptr boundary = boundary_clouds.at(it.first);

    // Pas the instance if there are not enough points.
    if(cloud->size() < param.min_points_of_cluster
       || boundary->size() < param.min_points_of_cluster){
      ROS_INFO("No enough clouds for instance %d", it.first);
      continue;
    }

    std::shared_ptr<unloader_msgs::Object> obj = std::make_shared<unloader_msgs::Object>();
    obj->type = unloader_msgs::Object::BOX;
    obj->instance_id = it.first;
    for(int i =0; i <6; i++)
      obj->visible_plane[i] = false;
    obj->point_cloud.header.frame_id = "robot";

    if(! ComputeBoxOBB(cloud, boundary, param, depth_dir, t_wc, obj, visualizer, floor_plane) )
      continue;

    // TODO Collect unsynced obb before matching.
    // The obj->instance_id is (yolact instance id == local id).
    // cam_lid_obj[cam_id][obj->instance_id] = obj;
  }
  return;
}

