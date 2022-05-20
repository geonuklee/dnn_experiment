#include "ros_util.h"
#include "ros_unet/SetCamera.h"
#include "ros_unet/ComputeFloor.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>

#include <g2o/types/slam3d/se3quat.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


pcl::PointCloud<pcl::PointXYZ>::Ptr Unproject(cv::Mat depth, cv::Mat nu_map, cv::Mat nv_map,
                                              cv::Mat init_mask,
                                              float voxel_leaf,
                                              std::function<bool(const pcl::PointXYZ&)> isInlier
                                              ){
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
      const float& nu = nu_map.at<float>(r,c);
      const float& nv = nv_map.at<float>(r,c);
      pcl::PointXYZ pt(nu*z, nv*z, z);
      if(isInlier(pt))
        cloud->push_back(pt);
    }
  }
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize (voxel_leaf, voxel_leaf, voxel_leaf);
  sor.filter(*cloud);
  return cloud;
}

Eigen::Vector4f RansacPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float threshold){
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  //seg.setMethodType(pcl::SAC_LMEDS);
  seg.setDistanceThreshold (threshold);
  seg.setInputCloud(cloud);
  seg.segment(*inliers, *coefficients);
  Eigen::Vector4f plane;
  for(int i =0; i <4; i++)
    plane[i] = coefficients->values.at(i);
  if(plane[2] > 0.)
    plane = -plane;
  return plane;
}

class FloorDetector{
public:
  FloorDetector(ros::NodeHandle& nh){
    pub_mask = nh.advertise<sensor_msgs::Image>("floor",1); 
  }

  bool SetCamera(ros_unet::SetCamera::Request  &req,
                 ros_unet::SetCamera::Response &res)
  {
    cv::Mat K = cv::Mat::zeros(3,3,CV_32F);
    for(int r=0; r < K.rows; r++)
      for(int c=0; c < K.cols; c++)
        K.at<float>(r,c) = req.info.K.data()[3*r+c];
    cv::Mat D = cv::Mat::zeros(req.info.D.size(),1,CV_32F);
    for (int j = 0; j < D.rows; j++)
      D.at<float>(j,0) = req.info.D.at(j);

    nu_map = cv::Mat::zeros(req.info.height, req.info.width, CV_32F);
    nv_map = cv::Mat::zeros(nu_map.rows, nu_map.cols, CV_32F);

    float fx = K.at<float>(0,0);
    float fy = K.at<float>(1,1);
    float cx = K.at<float>(0,2);
    float cy = K.at<float>(1,2);
    for(int r=0; r<nu_map.rows; r++){
      for(int c=0; c<nu_map.cols; c++){
        std::vector<cv::Point2f> vec_uv = { cv::Point2f(c,r) };
        std::vector<cv::Point2f> vec_nuv;
        cv::undistortPoints(vec_uv, vec_nuv, K, D);
        nu_map.at<float>(r,c) = vec_nuv.at(0).x;
        nv_map.at<float>(r,c) = vec_nuv.at(0).y;
      }
    }
    return true;
  }

  bool ComputeFloor(ros_unet::ComputeFloor::Request  &req,
                    ros_unet::ComputeFloor::Response &res)
  {
    cv::Mat init_mask = cv_bridge::toCvCopy(req.init_mask, sensor_msgs::image_encodings::TYPE_8UC1)->image;
    cv::Mat depth = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    float voxel_leaf = 0.02;
    auto f0 = [](const pcl::PointXYZ& pt){return true;};
    pcl::PointCloud<pcl::PointXYZ>::Ptr partial_cloud
      = Unproject(depth,nu_map,nv_map, init_mask, voxel_leaf,f0);
    // The coefficients are consist with Hessian normal form : [normal_x normal_y normal_z d].
    // ref : https://pointclouds.org/documentation/group__sample__consensus.html
    Eigen::Vector4f plane = RansacPlane(partial_cloud, 0.01);
    /*    
    const float margin1 = 0.1;
    auto f1 = [&plane,&margin1](const pcl::PointXYZ& pt){
      float d = plane.dot( Eigen::Vector4f(pt.x, pt.y, pt.z, 1.) );
      return  std::abs(d) < margin1;
    };
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = Unproject(depth,nu_map,nv_map,
                                                          req.y0*2,voxel_leaf,f1);
    const float margin2 = 0.005;
    plane = RansacPlane(cloud, 0.02);
    */
    const float margin2 = 0.005;
    auto f2 = [&plane, &margin2](const pcl::PointXYZ& pt){
      float d = plane.dot( Eigen::Vector4f(pt.x, pt.y, pt.z, 1.) );
      return  d < margin2;
    };

    cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    for(int r=0; r<depth.rows; r++){
      for(int c=0; c<depth.cols; c++){
        float z = depth.at<float>(r,c);
        if(z==0.)
          continue;
        const float& nu = nu_map.at<float>(r,c);
        const float& nv = nv_map.at<float>(r,c);
        if( init_mask.at<unsigned char>(r,c) )
          mask.at<unsigned char>(r,c) = 1;
        else{
          pcl::PointXYZ pt(nu*z, nv*z, z) ;
          if( f2(pt) )
            mask.at<unsigned char>(r,c) = 1;
        }
      }
    }

    cv_bridge::CvImage cv_img;
    if(pub_mask.getNumSubscribers() > 0){
      cv::Mat dst = cv_bridge::toCvCopy(req.rgb, sensor_msgs::image_encodings::TYPE_8UC3)->image;
      for(int r=0; r<dst.rows;r++){
        for(int c=0; c<dst.cols;c++){
          const unsigned char& m = mask.at<unsigned char>(r,c);
          const unsigned char& m0 = init_mask.at<unsigned char>(r,c);
          if(m && m0){
            auto& p = dst.at<cv::Vec3b>(r,c);
            p[1] = std::min<int>( (int) p[1] + 100, 255);
            for(int k=0; k<3; k++){
              if(k==1)
                continue;
              p[k] = std::max<int>( (int) p[k] - 50, 0);
            }
          }
          else if(m){
            auto& p = dst.at<cv::Vec3b>(r,c);
            p[0] = std::min<int>( (int) p[0] + 100, 255);
            for(int k=1; k<3; k++)
              p[k] = std::max<int>( (int) p[k] - 50, 0);
          }
        }
      }
      cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      cv_img.image    = dst;
      pub_mask.publish(cv_img.toImageMsg());
    }

    cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
    cv_img.image    = mask;
    cv_img.toImageMsg(res.mask);
    res.plane = {plane[0], plane[1], plane[2], plane[3]};
    return true;
  }
private:
  cv::Mat nu_map, nv_map;
  ros::Publisher pub_mask;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "~");
  ros::NodeHandle nh("~");
  FloorDetector floor_detector(nh);

  ros::ServiceServer s0
    = nh.advertiseService("SetCamera", &FloorDetector::SetCamera, &floor_detector);

  ros::ServiceServer s1
    = nh.advertiseService("ComputeFloor", &FloorDetector::ComputeFloor, &floor_detector);
  ros::spin();
  return 0;
}
