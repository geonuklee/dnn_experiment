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
    if(this->mask_.empty()){
      // TODO 2 channel-> mask_, convex edge 분리.
      cv::Mat mask_concave = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC2)->image;
      cv::Mat ch[2];
      cv::split(mask_concave,ch);
      this->mask_ = ch[0];
      this->convex_edge_ = ch[1];
    }
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
                                     cv::Mat& outline_edge,
                                     cv::Mat& convex_edge,
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
  outline_edge = rectified_mask==1;
  surebox_mask = rectified_mask==2;

  cv::remap(convex_edge_, convex_edge, map1_, map2_, cv::INTER_NEAREST);

  mask_ = cv::Mat();
  convex_edge_ = cv::Mat();
  return;
}
