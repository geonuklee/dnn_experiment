#include "ros_util.h"
#include "segment2d.h"
#include "mask2obb.h"

#include <vector>
#include <map>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>

#include <g2o/types/slam3d/se3quat.h>

void SubscribeTopic(const std::vector<int>& cameras,
                    ros::NodeHandle& nh,
                    std::map<std::string, ros::Subscriber>& subs,
                    std::map<int, cv::Mat>& rgbs,
                    std::map<int, cv::Mat>& depths,
                    std::map<int, sensor_msgs::CameraInfo>& camerainfos
                   ){
  for(int cam_id : cameras){
    std::string img_name   = "cam"+ std::to_string(cam_id) +"/rgb";
    std::string depth_name = "cam"+ std::to_string(cam_id) +"/depth";
    std::string info_name  = "cam"+ std::to_string(cam_id) +"/camera_info";

    auto img_callback = [cam_id, &rgbs](const sensor_msgs::ImageConstPtr& msg){
      rgbs[cam_id] = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
    };
    subs[img_name] = nh.subscribe<sensor_msgs::ImageConstPtr, const sensor_msgs::ImageConstPtr& >(img_name, 1, img_callback);

    ROS_INFO("Subscribe %s",subs[img_name].getTopic().c_str());

    auto depth_callback = [cam_id, &depths](const sensor_msgs::ImageConstPtr& msg){
      cv::Mat depth = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
      double minVal, maxVal;
      cv::minMaxLoc( depth, &minVal, &maxVal);
      // Convert unit to [meter]
      if(maxVal > 1000.)
        depth /= 1000.;
      depths[cam_id] = depth;

    };

    ROS_INFO("Subscribe %s",subs[depth_name].getTopic().c_str());
    subs[depth_name] = nh.subscribe<sensor_msgs::ImageConstPtr, const sensor_msgs::ImageConstPtr& >(depth_name, 1, depth_callback);

    auto info_callback = [cam_id, &camerainfos](const sensor_msgs::CameraInfoConstPtr& msg){
      camerainfos[cam_id] = *msg;
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
                  const std::map<std::string, ros::Subscriber>& subs,
                  std::map<int, cv::Mat>& rgbs,
                  std::map<int, cv::Mat>& depths,
                  std::map<int, sensor_msgs::CameraInfo>& camerainfos
                  ){
  rgbs.clear();
  depths.clear();
  ros::Rate rate(10);
  while( !ros::isShuttingDown() ){
    bool ready = true;
    for(int cam_id : cameras){
      if(!rgbs.count(cam_id)){
        std::string img_name   = "cam"+ std::to_string(cam_id) +"/rgb";
        ROS_DEBUG("No topic for %s", subs.at(img_name).getTopic().c_str() );
        ready = false;
      }
      else if(!depths.count(cam_id)){
        std::string depth_name = "cam"+ std::to_string(cam_id) +"/depth";
        ROS_DEBUG("No topic for %s", subs.at(depth_name).getTopic().c_str() );
        ready = false;
      }
      else if(!camerainfos.count(cam_id)){
        std::string info_name  = "cam"+ std::to_string(cam_id) +"/camera_info";
        ROS_DEBUG("No topic for %s", subs.at(info_name).getTopic().c_str() );
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


ObbParam GetObbParam(ros::NodeHandle& nh){
  // Get rosparams.
  ObbParam param;
  std::vector<int> cameras = GetRequiredParamVector<int>(nh, "cameras");
  param.min_z_floor = GetRequiredParam<double>(nh, "min_z_floor");
  param.min_points_of_cluster = GetRequiredParam<double>(nh, "min_points_of_cluster");
  param.voxel_leaf = GetRequiredParam<double>(nh, "voxel_leaf");
  param.verbose = GetRequiredParam<bool>(nh, "verbose");

  {
    int match_mode = GetRequiredParam<int>(nh, "match_mode");
    if(match_mode < 0 || match_mode > 2){
      ROS_FATAL("match_mode of mask2obb is out of range");
      exit(1);
    }
    param.match_mode = static_cast<ObbParam::MATCHMODE>( match_mode );
  }

  {
    int match_method = GetRequiredParam<int>(nh, "match_method");
    if(match_method < 0 || match_method > 1){
      ROS_FATAL("match_method of mask2obb is out of range");
      exit(1);
    }
    param.match_method = static_cast<ObbParam::MATCHMETHOD>( match_method );
  }

  {
    std::string sensor_model = GetRequiredParam<std::string>(nh, "sensor_model");
    if(sensor_model == "helios")
      param.sensor_model = ObbParam::SENSORMODEL::HELIOS;
    else if(sensor_model == "k4a")
      param.sensor_model = ObbParam::SENSORMODEL::K4A;
    else{
      ROS_ERROR_ONCE("Unexpected sensor model for mask2obb");
      throw 1;
    }
  }
  return param;
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
  bool generate_points = GetRequiredParam<bool>(nh, "generate_points");

  ObbParam param = GetObbParam(nh);
  const bool verbose_segment2d = true;

  std::map<std::string, ros::Subscriber> subs;
  std::map<int, cv::Mat> rgbs;
  std::map<int, cv::Mat> depths;
  std::map<int, sensor_msgs::CameraInfo> camerainfos;
  SubscribeTopic(cameras, nh, subs, rgbs, depths, camerainfos);
  std::map<int, std::shared_ptr<Segment2DAbstract > > segment2d;
  std::map<int, std::shared_ptr<ObbEstimator> > obb_estimators;

  std::map<int, ros::Publisher> pub_clouds, pub_boundary, pub_vis_mask;
  ros::Publisher pub_xyzrgb;
  if(generate_points)
    pub_xyzrgb = nh.advertise<sensor_msgs::PointCloud2>("xyzrgb",1);

  UpdateTopics(cameras, nh, subs, rgbs, depths, camerainfos);
  for(int cam_id : cameras){
    std::string cam_name = "cam"+std::to_string(cam_id);
    segment2d[cam_id] = std::make_shared<Segment2DEdgeSubscribe>(camerainfos.at(cam_id),
                                                                 compute_size,
                                                                 cam_name,
                                                                 nh
                                                                );
    obb_estimators[cam_id] = std::make_shared<ObbEstimator>(camerainfos.at(cam_id) );
    pub_clouds[cam_id]   = nh.advertise<sensor_msgs::PointCloud2>(cam_name+"/clouds",1);
    pub_boundary[cam_id] = nh.advertise<sensor_msgs::PointCloud2>(cam_name+"/boundary",1);
    pub_vis_mask[cam_id] = nh.advertise<sensor_msgs::Image>(cam_name+"/vis_mask",1);
  }

  std::map<int, MarkerCamera > marker_cameras;
  EigenMap<int, g2o::SE3Quat > Tcws;

  // Publishers for process visualization.
  std::map<int, std::shared_ptr<ObbProcessVisualizer> > obb_process_visualizers;
  for(int cam_id : cameras){
    auto visualizer = std::make_shared<ObbProcessVisualizer>(cam_id, nh);
    obb_process_visualizers[cam_id] = visualizer;
  }


  ros::Rate rate(2);
  while(!ros::isShuttingDown()){
    UpdateTcws(cameras, nh, Tcws);
    UpdateTopics(cameras, nh, subs, rgbs, depths, camerainfos);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb;
    if(generate_points){
      xyzrgb = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>() );
      xyzrgb->points.reserve( cameras.size() * compute_size.width* compute_size.height);
    }

    for(int cam_id : cameras){
      std::map<int,int> ins2cls;
      cv::Mat instance_marker, edge_distance, depthmap;
      auto segmenter = segment2d.at(cam_id);
      bool b = segmenter->Process(rgbs.at(cam_id),
                                  depths.at(cam_id),
                                  instance_marker, edge_distance, depthmap,
                                  ins2cls, verbose_segment2d
                                 );
      if(!b)
        break;
      std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> segmented_clouds, boundary_clouds;
      obb_estimators.at(cam_id)->GetSegmentedCloud(Tcws.at(cam_id),
                                                   segmenter->GetRectifiedRgb(),
                                                   segmenter->GetRectifiedDepth(),
                                                   instance_marker,
                                                   param,
                                                   segmented_clouds, boundary_clouds, xyzrgb);
      // TODO Compute OBB for each instance
      obb_estimators.at(cam_id)->ComputeObbs(segmented_clouds,
                                             boundary_clouds,
                                             param,
                                             Tcws.at(cam_id),
                                             cam_id,
                                             obb_process_visualizers.at(cam_id)
                                             );

      for(auto it_visualizer : obb_process_visualizers)
        it_visualizer.second->Visualize();

      // TODO matching???

      if(pub_vis_mask.at(cam_id).getNumSubscribers() > 0){
        cv::Mat rect_rgb = segmenter->GetRectifiedRgb();
        cv::Mat dst = Overlap(rect_rgb, instance_marker);
        cv_bridge::CvImage msg;
        msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
        msg.image    = dst;
        pub_vis_mask[cam_id].publish(msg.toImageMsg());
      }
      if(pub_clouds.at(cam_id).getNumSubscribers() > 0){
        sensor_msgs::PointCloud2 msg;
        ColorizeSegmentation(segmented_clouds, msg);
        pub_clouds.at(cam_id).publish(msg);
      }
      if(pub_boundary.at(cam_id).getNumSubscribers() > 0){
        sensor_msgs::PointCloud2 msg;
        ColorizeSegmentation(boundary_clouds, msg);
        pub_boundary.at(cam_id).publish(msg);
      }
    }

    if(generate_points) {
      sensor_msgs::PointCloud2 msg;
      pcl::toROSMsg(*xyzrgb, msg);
      msg.header.frame_id = "robot";
      pub_xyzrgb.publish(msg);
    }

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
