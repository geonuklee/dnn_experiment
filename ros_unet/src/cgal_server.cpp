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
//#include "ros_unet/ClearCamera.h"
//#include "ros_unet/SetCamera.h"
#include "ros_unet/ComputePoints2Obb.h"
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
  BoxDetector(ros::NodeHandle& nh)
  : nh_(nh){
    pub_clouds_ = nh_.advertise<sensor_msgs::PointCloud2>("clouds",1);
    pub_poses_  = nh_.advertise<geometry_msgs::PoseArray>("poses",1);
    pub_obbs_   = nh_.advertise<visualization_msgs::MarkerArray>("obbs",1);
  }

  bool ComputeObb(ros_unet::ComputePoints2Obb::Request &req,
                  ros_unet::ComputePoints2Obb::Response &res)
  {
    std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
    std::map<int, size_t> n_points;
    assert(req.xyz_clouds.size() == 3*req.l_clouds.size() );
    for(size_t i =0; i < req.l_clouds.size(); i++){
      const int l = req.l_clouds[i];
      n_points[l]++;
    }
    for(auto it : n_points){
      auto ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
      ptr->reserve(it.second);
      clouds[it.first] = ptr;
    }
    {
      float* data = req.xyz_clouds.data();
      for(size_t i =0; i < req.l_clouds.size(); i++){
        const int l = req.l_clouds[i];
        float* xyz = data+3*i;
        clouds[l]->push_back(pcl::PointXYZ(xyz[0],xyz[1],xyz[2]) );
        //Eigen::Vector3d Xw = Twc * Eigen::Vector3d(xyz[0],xyz[1],xyz[2]);
        //clouds[l]->push_back(pcl::PointXYZ(Xw.x(),Xw.y(),Xw.z()));
      }
    }

    geometry_msgs::PoseArray poses_array;
    visualization_msgs::MarkerArray obbs;
    poses_array.header.frame_id = req.frame_id;

    for(auto it_cloud : clouds){
      const int& id = it_cloud.first;
      const auto ptr = it_cloud.second;
      //printf("n( #%d) = %ld\n", id, ptr->size() );
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
      pub_poses_.publish(poses_array);
    }
    pub_obbs_.publish(obbs);

    if(pub_clouds_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      ColorizeSegmentation(clouds, msg);
      pub_clouds_.publish(msg);
    }

    return true;
  }


private:

  g2o::SE3Quat GetSE3Quat(const geometry_msgs::Pose& pose) const {
    const auto& p = pose.position;
    const auto& q = pose.orientation;
    Eigen::Quaterniond quat(q.w,q.x,q.y,q.z);
    Eigen::Vector3d t(p.x,p.y,p.z);
    g2o::SE3Quat se3quat(quat.toRotationMatrix(), t);
    return se3quat;
  }

  ros::NodeHandle& nh_;
  ros::Publisher pub_clouds_;
  ros::Publisher pub_poses_;
  ros::Publisher pub_obbs_;
};

int main(int argc, char** argv){
  ros::init(argc, argv, "~");
  ros::NodeHandle nh("~");
  BoxDetector box_detector(nh);
  ros::ServiceServer s
    = nh.advertiseService("ComputeObb", &BoxDetector::ComputeObb, &box_detector);
  ros::spin();
  return 0;
}
