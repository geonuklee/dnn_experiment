#include "ros_util.h"
#include "segment2d.h"

#include <vector>
#include <map>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <g2o/types/slam3d/se3quat.h>

void SubscribeTopic(const std::vector<int>& cameras,
                    ros::NodeHandle& nh,
                    std::map<std::string, ros::Subscriber>& subs,
                    std::map<int, sensor_msgs::ImageConstPtr>& rgb_ptr,
                    std::map<int, sensor_msgs::ImageConstPtr>& depth_ptr,
                    std::map<int, sensor_msgs::CameraInfo>& camera_info
                   ){
  for(int cam_id : cameras){
    std::string img_name   = "cam"+ std::to_string(cam_id) +"/rgb";
    std::string depth_name = "cam"+ std::to_string(cam_id) +"/depth";
    std::string info_name  = "cam"+ std::to_string(cam_id) +"/camera_info";

    auto img_callback = [cam_id, &rgb_ptr](const sensor_msgs::ImageConstPtr& msg){
      rgb_ptr[cam_id] = msg;
    };
    subs[img_name] = nh.subscribe<sensor_msgs::ImageConstPtr, const sensor_msgs::ImageConstPtr& >(img_name, 1, img_callback);

    ROS_INFO("Subscribe %s",subs[img_name].getTopic().c_str());

    auto depth_callback = [cam_id, &depth_ptr](const sensor_msgs::ImageConstPtr& msg){
      depth_ptr[cam_id] = msg;
    };

    ROS_INFO("Subscribe %s",subs[depth_name].getTopic().c_str());
    subs[depth_name] = nh.subscribe<sensor_msgs::ImageConstPtr, const sensor_msgs::ImageConstPtr& >(depth_name, 1, depth_callback);

    auto info_callback = [cam_id, &camera_info](const sensor_msgs::CameraInfoConstPtr& msg){
      camera_info[cam_id] = *msg;
    };

    ROS_INFO("Subscribe %s",subs[info_name].getTopic().c_str());
    subs[info_name] = nh.subscribe<sensor_msgs::CameraInfoPtr, const sensor_msgs::CameraInfoConstPtr& >(info_name, 1, info_callback);

  }
  return;
}


void UpdateTcws(const std::vector<int>& cameras,
               ros::NodeHandle& nh,
               EigenMap<int, g2o::SE3Quat >& Tcws){
  Tcws.clear();

  ros::Rate rate(10);
  while(true){
    for(int i : cameras){
      std::string name = "/cam"+std::to_string(i)+"/base_T_cam";
      std::vector<double> vec;
      if( !nh.getParam(name, vec) ){
        ROS_WARN("No rosparam for %s", name.c_str());
        break;
      }
      Eigen::Matrix<double,4,4> RTrc;
      for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
          RTrc(i,j) = vec.at(i*4 + j);
      g2o::SE3Quat Tcr(RTrc.block<3,3>(0,0).transpose(),
                       -RTrc.block<3,3>(0,0).transpose()*RTrc.block<3,1>(0,3) );
      Tcws[i] = Tcr;
    }
    if(Tcws.size() == cameras.size())
      break;
    rate.sleep();
    ros::spinOnce();
  }
  return;
}

void UpdateTopics(const std::vector<int>& cameras,
                  ros::NodeHandle& nh,
                  std::map<int, sensor_msgs::ImageConstPtr>& rgb_ptr,
                  std::map<int, sensor_msgs::ImageConstPtr>& depth_ptr,
                  std::map<int, sensor_msgs::CameraInfo>& camerainfo_ptr
                  ){
  rgb_ptr.clear();
  depth_ptr.clear();
  ros::Rate rate(10);
  while( !ros::isShuttingDown() ){
    bool ready = true;
    for(int cam_id : cameras){
      if(!rgb_ptr.count(cam_id)){
        ROS_DEBUG("No rgb for cam %d", cam_id );
        ready = false;
      }
      else if(!depth_ptr.count(cam_id)){
        ROS_DEBUG("No depth for cam %d", cam_id );
        ready = false;
      }
      else if(!camerainfo_ptr.count(cam_id)){
        ROS_DEBUG("No camera_info for cam %d", cam_id );
        ready = false;
      }
      if(!ready)
        break;
    }
    if(!ready){
      ros::spinOnce();
      rate.sleep();
    }
    else
      break;
  }
  return;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "pipeline");
  ros::NodeHandle nh("~");
  std::vector<int> cameras = GetRequiredParamVector<int>(nh, "cameras");
  cv::Size compute_size; {
    std::vector<int> compute_image_size = GetRequiredParamVector<int>(nh, "compute_image_size");
    compute_size.width = compute_image_size[0];
    compute_size.height = compute_image_size[1];
  }

  const bool verbose_segment2d = true;

  std::map<std::string, ros::Subscriber> subs;
  std::map<int, sensor_msgs::ImageConstPtr> rgb_ptr;
  std::map<int, sensor_msgs::ImageConstPtr> depth_ptr;
  std::map<int, sensor_msgs::CameraInfo> camerainfo_ptr;
  SubscribeTopic(cameras, nh, subs, rgb_ptr,depth_ptr,camerainfo_ptr);
  std::map<int, std::shared_ptr<Segment2DAbstract > > segment2d;

  UpdateTopics(cameras, nh, rgb_ptr, depth_ptr, camerainfo_ptr);
  for(int cam_id : cameras){
    std::string cam_name = "cam"+std::to_string(cam_id);
#if 1
    segment2d[cam_id] = std::make_shared<Segment2DEdgeSubscribe>(camerainfo_ptr.at(cam_id),
                                                                 compute_size,
                                                                 cam_name,
                                                                 nh
                                                                );
#else
    segment2d[cam_id] = std::make_shared<Segment2Dthreshold>(camerainfo_ptr.at(cam_id),
                                                             compute_size,
                                                             cam_name,
                                                             -0.06);
#endif
  }

  std::map<int, MarkerCamera > marker_cameras;
  EigenMap<int, g2o::SE3Quat > Tcws;

  ros::Rate rate(2);
  while(!ros::isShuttingDown()){
    UpdateTcws(cameras, nh, Tcws);
    UpdateTopics(cameras, nh, rgb_ptr, depth_ptr, camerainfo_ptr);

    bool segment_is_done = true;
    for(int cam_id : cameras){
      std::map<int,int> ins2cls;
      cv::Mat instance_marker, edge_distance, depthmap;
      segment_is_done &= segment2d.at(cam_id)->Process(rgb_ptr.at(cam_id),
                                                       depth_ptr.at(cam_id),
                                                       instance_marker, edge_distance, depthmap,
                                                       ins2cls, verbose_segment2d
                                                      );
    }
    if(!segment_is_done)
      continue;

    ROS_INFO("Segment");
    if(verbose_segment2d){
      char c = cv::waitKey(1);
      if(c == 'q')
        break;
    }
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
