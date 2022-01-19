#ifndef SEGMENT2D_H_
#define SEGMENT2D_H_

#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

struct MarkerCamera{
  MarkerCamera() { }
  MarkerCamera(const cv::Mat& K,
               const cv::Mat& D,
               const cv::Size& image_size);

  cv::Mat K_;
  cv::Mat D_;
  cv::Size image_size_;
};

class Segment2DAbstract {
public:

  Segment2DAbstract(const sensor_msgs::CameraInfo& camera_info, cv::Size compute_size, std::string name);

  virtual bool Process(sensor_msgs::Image::ConstPtr given_rgb,
               sensor_msgs::Image::ConstPtr given_depth,
               cv::Mat& marker, cv::Mat& edge_distance, cv::Mat& depthmap,
               std::map<int,int>& instance2class,
               bool verbose=false) = 0;

  const MarkerCamera GetComputeIntrinsic() const { return camera_; }

  void Rectify(sensor_msgs::Image::ConstPtr given_rgb,
               sensor_msgs::Image::ConstPtr given_depth,
               cv::Mat& rectified_rgb,
               cv::Mat& depthmap
              ) const;

protected:
  MarkerCamera camera_;
  const std::string name_;
  cv::Mat map1_, map2_;
};

class Segment2DEdgeBased : public Segment2DAbstract{
public:
  Segment2DEdgeBased(const sensor_msgs::CameraInfo& camerainfo,
                     cv::Size compute_size,
                     std::string name
                     );

  virtual bool Process(sensor_msgs::Image::ConstPtr given_rgb,
                       sensor_msgs::Image::ConstPtr given_depth,
                       cv::Mat& marker, cv::Mat& edge_distance, cv::Mat& depthmap,
                       std::map<int,int>& instance2class,
                       bool verbose=false);

  bool Process(cv::Mat rgb, cv::Mat depth,
               cv::Mat& marker, cv::Mat& edge_distance,
               std::map<int,int>& instance2class,
               bool verbose=false);

  virtual cv::Mat GetEdge(const cv::Mat rgb, const cv::Mat depth, const cv::Mat validmask,
                          bool verbose) = 0;
protected:

  cv::Mat vignett32S_;
  cv::Mat vignett8U_;
};

class Segment2Dthreshold : public Segment2DEdgeBased {
public:
  Segment2Dthreshold(const sensor_msgs::CameraInfo& camerainfo,
                     cv::Size compute_size,
                     std::string name,
                     double lap_depth_threshold
                    );
protected:
  virtual cv::Mat GetEdge(const cv::Mat rgb, const cv::Mat depth, const cv::Mat validmask,
                          bool verbose);

  const double lap_depth_threshold_;
};


#include <ros/ros.h>
class Segment2DEdgeSubscribe : public Segment2DEdgeBased {
public:
  Segment2DEdgeSubscribe(const sensor_msgs::CameraInfo& camerainfo,
                         cv::Size compute_size,
                         std::string name,
                         ros::NodeHandle& nh
                        );

  virtual ~Segment2DEdgeSubscribe();
protected:
  virtual cv::Mat GetEdge(const cv::Mat rgb, const cv::Mat depth, const cv::Mat validmask,
                          bool verbose);

  ros::Subscriber edge_subscriber_;
  cv::Mat edge_;
};




cv::Mat GetDepthMask(const cv::Mat depth);
cv::Mat GetEdgeMask(const cv::Mat depthmask);

cv::Mat GetGroove(const cv::Mat marker,
                  const cv::Mat depth,
                  const cv::Mat validmask,
                  const cv::Mat vignett,
                  int bg_idx);


#endif
