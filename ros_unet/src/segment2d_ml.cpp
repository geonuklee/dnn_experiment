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
    this->edge_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1)->image;
  };

  std::string img_name = name+"/edge";
  edge_subscriber_ = nh.subscribe<sensor_msgs::ImageConstPtr,
                   const sensor_msgs::ImageConstPtr& >(img_name, 1, img_callback);

}

Segment2DEdgeSubscribe::~Segment2DEdgeSubscribe() {

}

cv::Mat Segment2DEdgeSubscribe::GetEdge(const cv::Mat rgb, const cv::Mat depth, const cv::Mat validmask,
                          bool verbose)
{
  cv::Mat edge;
  ros::Rate rate(100);
  int ntry = 0;
  while(!ros::isShuttingDown()){
    if(!edge_.empty())
      break;
    if(++ntry%100 == 0){
      std::cout << edge_subscriber_.getTopic()  << std::endl;
      ROS_WARN("No topic for %s", edge_subscriber_.getTopic().c_str() );
    }
    rate.sleep();
    ros::spinOnce();
  }
  edge = edge_.clone();
  edge_ = cv::Mat();
  return edge;
}
