#include <ros/ros.h>
#include <vector>
#include <map>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>
#include <g2o/types/slam3d/se3quat.h>

#include "ros_util.h"
#include "ros_unet/ClearCamera.h"
#include "ros_unet/SetCamera.h"
#include "ros_unet/ComputeCgalObb.h"
#include "mask2obb.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Aff_transformation_3.h>
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

namespace PMP = CGAL::Polygon_mesh_processing;
typedef CGAL::Exact_predicates_inexact_constructions_kernel    CGAL_KERNEL;
typedef CGAL_KERNEL::Point_3                                   Point;
typedef CGAL::Surface_mesh<Point>                              Surface_mesh;

class BoxDetector {
public:
  BoxDetector(ros::NodeHandle& nh, const ObbParam& param)
  : nh_(nh), param_(param) {
  }

  bool ClearCamera(ros_unet::ClearCamera::Request  &req,
                 ros_unet::ClearCamera::Response &res){

    return true;
  }

  bool SetCamera(ros_unet::SetCamera::Request  &req,
                 ros_unet::SetCamera::Response &res)
  {
    const std::vector<double>& D = req.info.D;
    double v_max = 0.;
    for(const double& v : D)
      v_max = std::max<double>(std::abs<double>(v), v_max);

    MarkerCamera camera;
    if(v_max > 1e-5){
      assert(false); // No support yet.
      // TODO newK with cv::getOptimalNewCamera
    } else {
      std::cout << "No distortion " << std::endl;
      cv::Mat K = cv::Mat::zeros(3,3,CV_32F);
      for(int i = 0; i<K.rows; i++)
        for(int j = 0; j < K.cols; j++)
          K.at<float>(i,j) = req.info.K.data()[3*i+j];
      cv::Mat D = cv::Mat::zeros(req.info.D.size(),1,CV_32F);
      for (int j = 0; j < D.rows; j++)
        D.at<float>(j,0) = req.info.D.at(j);
      cv::Size osize(req.info.width, req.info.height);
      camera = MarkerCamera(K,D,osize);
    }

    cv::Mat cvK = camera.K_;
    cv::Mat cvD = camera.D_;
    nu_ = cv::Mat::zeros(camera.image_size_.height, camera.image_size_.width,CV_32F);
    nv_ = cv::Mat::zeros(camera.image_size_.height, camera.image_size_.width,CV_32F);
    for(int r = 0; r < nu_.rows; r++){
      for(int c = 0; c < nu_.cols; c++){
        cv::Point2f uv;
        uv.x = c;
        uv.y = r;
        std::vector<cv::Point2f> vec_uv = {uv};
        pcl::PointXYZ xyz;
        std::vector<cv::Point2f> normalized_pt;
        cv::undistortPoints(vec_uv, normalized_pt, cvK, cvD);
        nu_.at<float>(r,c) = normalized_pt[0].x;
        nv_.at<float>(r,c) = normalized_pt[0].y;
      }
    }

    const std::string& cam_id = req.cam_id.data;
    pub_clouds_[cam_id] = nh_.advertise<sensor_msgs::PointCloud2>(cam_id+"/clouds",1);
    pub_poses_[cam_id]  = nh_.advertise<geometry_msgs::PoseArray>(cam_id+"/poses",1);
    pub_obbs_[cam_id]   = nh_.advertise<visualization_msgs::MarkerArray>(cam_id+"/obbs",1);
    return true;
  }

  bool ComputeCgalObb(ros_unet::ComputeCgalObb::Request &req,
                  ros_unet::ComputeCgalObb::Response &res)
  {
    const std::string& cam_id = req.cam_id.data;
    g2o::SE3Quat Twc = GetSE3Quat(req.Twc);
    g2o::SE3Quat Tcw = Twc.inverse();
    cv::Mat depth, marker;
    GetCvMat(req, depth, marker);
    std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;

    const float max_depth = 5.; // TODO Paramterize
    const int boundary = 5;
    const int boundary2 = boundary*2;
    const int frame_offset = 30;
    const float f_boundary = boundary;
    int max_idx = 0;
    for(int r = boundary; r < depth.rows-boundary; r++){
      for(int c = boundary; c < depth.cols-boundary; c++){
        bool near_frame_boundary
          = r < frame_offset ||
          r > depth.rows-frame_offset ||
          c < frame_offset ||
          c > depth.cols-frame_offset;
        float z0 = depth.at<float>(r,c);
        if(z0 < 0.000001 || z0 > max_depth)
          continue;
        const int& idx = marker.at<int>(r,c);
        if(idx == 0)
          continue;
        if(!clouds.count(idx)){
          pcl::PointCloud<pcl::PointXYZ>::Ptr ptr_cloud(new pcl::PointCloud<pcl::PointXYZ>());
          ptr_cloud->reserve(1000);
          clouds[idx] = ptr_cloud;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr ptr = clouds.at(idx);
        pcl::PointXYZ xyz;
        cv::Point2f nuv = this->GetUV(r,c);
        Eigen::Vector3d Xc(nuv.x*z0, nuv.y*z0,z0);
        Eigen::Vector3d Xw = Twc*Xc;
        xyz.x = Xw[0];
        xyz.y = Xw[1];
        xyz.z = Xw[2];
        ptr->push_back(xyz);
      }
    }

    std::set<int> erase_list;
    for(auto it_cloud : clouds){
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(it_cloud.second);
      sor.setLeafSize (param_.voxel_leaf, param_.voxel_leaf, param_.voxel_leaf);
      sor.filter(*it_cloud.second);

      pcl::PointIndices::Ptr indices = EuclideanFilterXYZ(it_cloud.second, param_, true);
      if(indices->indices.empty()){
        erase_list.insert(it_cloud.first);
        continue;
      }
      pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(it_cloud.second);
      extract.setIndices(indices);
      extract.setNegative(false);
      extract.filter(*filtered_cloud);
      clouds[it_cloud.first] = filtered_cloud;
    }

    // TODO Compute OBB
    geometry_msgs::PoseArray poses_array;
    visualization_msgs::MarkerArray obbs;
    poses_array.header.frame_id = "robot";

    for(auto it_cloud : clouds){
      const int& id = it_cloud.first;
      const auto ptr = it_cloud.second;
      if(ptr->points.size() < 10)
        continue;
      std::vector<Point> points;
      points.reserve(ptr->points.size());
      for(const auto pt : ptr->points)
         points.push_back(Point(pt.x,pt.y,pt.z));
      CGAL_KERNEL::Aff_transformation_3 aff_tf;
      CGAL::oriented_bounding_box(points, aff_tf);
      Eigen::Matrix<double,3,3> R0w;
      for(int r=0; r<3; r++){
        for(int c=0; c<3; c++){
          R0w(r,c) = aff_tf.m(r,c);
        }
      }
      Eigen::Vector3d min_x(999.,999.,999.);
      Eigen::Vector3d max_x = -min_x;
      for(const auto _pt : ptr->points){
        Eigen::Vector3d pt = Eigen::Vector3d(_pt.x,_pt.y,_pt.z);
        Eigen::Vector3d pt0 = R0w*pt;
        for(int i=0; i<3; i++){
          min_x[i] = std::min(min_x[i],pt0[i]);
          max_x[i] = std::max(max_x[i],pt0[i]);
        }
      }
      Eigen::Vector3d cp = 0.5*R0w.transpose()*(max_x+min_x);
      Eigen::Vector3d scale0 = max_x-min_x;
 
      Eigen::Vector3d scale = scale0;
      g2o::SE3Quat Twb(R0w.transpose(), cp);
      FitAxis(Twb, scale);

      geometry_msgs::Pose center_pose;
      center_pose.position.x = Twb.translation().x();
      center_pose.position.y = Twb.translation().y();
      center_pose.position.z = Twb.translation().z();
      center_pose.orientation.w = Twb.rotation().w();
      center_pose.orientation.x = Twb.rotation().x();
      center_pose.orientation.y = Twb.rotation().y();
      center_pose.orientation.z = Twb.rotation().z();
      poses_array.poses.push_back(center_pose);
      {
        visualization_msgs::Marker marker;
        marker.id = id;
        marker.header.frame_id = poses_array.header.frame_id;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.pose = center_pose;
        marker.color = GetColor(id);
        marker.scale.x = scale.x();
        marker.scale.y = scale.y();
        marker.scale.z = scale.z();
        obbs.markers.push_back(marker);
      }
    }
    res.output = obbs;
    {
      visualization_msgs::Marker a;
      a.action = visualization_msgs::Marker::DELETEALL;
      obbs.markers.push_back(a);
      std::reverse(obbs.markers.begin(), obbs.markers.end());
    }
  
    if(poses_array.poses.size() > 0){
      pub_poses_[cam_id].publish(poses_array);
      pub_obbs_[cam_id].publish(obbs);
    }

    if(pub_clouds_.at(cam_id).getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      ColorizeSegmentation(clouds, msg);
      pub_clouds_.at(cam_id).publish(msg);
    }

    return true;
  }


private:

  cv::Point2f GetUV(int r, int c) const {
    cv::Point2f nuv;
    nuv.x = nu_.at<float>(r,c);
    nuv.y = nv_.at<float>(r,c);
    return nuv;
  }

  void GetCvMat(ros_unet::ComputeCgalObb::Request& req,
                cv::Mat& depth,
                cv::Mat& marker
                ) {
    depth = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    marker = cv_bridge::toCvCopy(req.marker, sensor_msgs::image_encodings::TYPE_32SC1)->image;
    return;
  }

  g2o::SE3Quat GetSE3Quat(const geometry_msgs::Pose& pose) const {
    const auto& p = pose.position;
    const auto& q = pose.orientation;
    Eigen::Quaterniond quat(q.w,q.x,q.y,q.z);
    Eigen::Vector3d t(p.x,p.y,p.z);
    g2o::SE3Quat se3quat(quat.toRotationMatrix(), t);
    return se3quat;
  }

  ObbParam param_;
  ros::NodeHandle& nh_;
  std::map<std::string, ros::Publisher> pub_clouds_;
  std::map<std::string, ros::Publisher> pub_poses_;
  std::map<std::string, ros::Publisher> pub_obbs_;
  cv::Mat nu_, nv_;
};

int main(int argc, char** argv){
  ros::init(argc, argv, "~");
  ros::NodeHandle nh("~");
  ObbParam param;
  param.min_points_of_cluster = GetRequiredParam<double>(nh, "min_points_of_cluster");
  param.voxel_leaf = GetRequiredParam<double>(nh, "voxel_leaf");
  param.euclidean_filter_tolerance = GetRequiredParam<double>(nh, "euc_tolerance"); // [meter] small tolerance condisering bg invasion
  param.verbose = GetRequiredParam<bool>(nh, "verbose");

  BoxDetector box_detector(nh, param);
  ros::ServiceServer s0
    = nh.advertiseService("SetCamera", &BoxDetector::SetCamera, &box_detector);
  ros::ServiceServer s1
    = nh.advertiseService("ClearCamera", &BoxDetector::ClearCamera, &box_detector);
  ros::ServiceServer s2
    = nh.advertiseService("ComputeCgalObb", &BoxDetector::ComputeCgalObb, &box_detector);


  ros::spin();
  return 0;
}
