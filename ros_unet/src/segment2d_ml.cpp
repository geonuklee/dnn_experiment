#include "segment2d.h"
#include <cv_bridge/cv_bridge.h>

Segment2DEdgeSubscribe::Segment2DEdgeSubscribe(const sensor_msgs::CameraInfo& camerainfo,
                                               cv::Size compute_size,
                                               std::string name,
                                               ros::NodeHandle& nh
                                              )
: Segment2DEdgeBased(camerainfo, compute_size, name)
{

  auto img_callback = [this](const sensor_msgs::ImageConstPtr& msg){
    if(this->mask_.empty())
      this->mask_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1)->image;
  };

  std::string img_name = name+"/mask";
  edge_subscriber_ = nh.subscribe<sensor_msgs::ImageConstPtr,
                   const sensor_msgs::ImageConstPtr& >(img_name, 1, img_callback);

}

Segment2DEdgeSubscribe::~Segment2DEdgeSubscribe() {

}

void Segment2DEdgeSubscribe::GetEdge(const cv::Mat rgb,
                                     const cv::Mat depth,
                                     const cv::Mat validmask,
                                     cv::Mat& edge,
                                     cv::Mat& surebox_mask,
                                     bool verbose)
{
  ros::Rate rate(100);
  int ntry = 0;
  while(!ros::isShuttingDown()){
    if(!mask_.empty())
      break;
    if(++ntry%100 == 0){
      ROS_WARN("No topic for %s", edge_subscriber_.getTopic().c_str() );
    }
    rate.sleep();
    ros::spinOnce();
  }

  cv::Mat rectified_mask;
  cv::remap(mask_, rectified_mask, map1_, map2_, cv::INTER_NEAREST);
  edge = rectified_mask==1;
  surebox_mask = rectified_mask==2;
  mask_ = cv::Mat();
  return;
}
