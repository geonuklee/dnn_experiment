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

  sensor_msgs::CameraInfo AsCameraInfo() const;

  cv::Mat K_;
  cv::Mat D_;
  cv::Size image_size_;
};

class Segment2DAbstract {
public:

  Segment2DAbstract(const sensor_msgs::CameraInfo& camera_info, cv::Size compute_size, std::string name);

  virtual bool Process(const cv::Mat given_rgb,
                       const cv::Mat given_depth,
                       cv::Mat& marker,
                       cv::Mat& convex_edge,
                       cv::Mat& depthmap,
                       std::map<int,int>& instance2class,
                       bool verbose=false) = 0;

  const MarkerCamera& GetRectifiedIntrinsic() const { return camera_; }
  cv::Mat GetRectifiedRgb() const { return rectified_rgb_; }
  cv::Mat GetRectifiedDepth() const { return rectified_depth_; }
  
protected:
  void Rectify(cv::Mat given_rgb,
               cv::Mat given_depth
              );

  MarkerCamera camera_;
  const std::string name_;
  cv::Mat map1_, map2_;

  cv::Mat rectified_rgb_;
  cv::Mat rectified_depth_;
};

class Segment2DEdgeBased : public Segment2DAbstract{
public:
  Segment2DEdgeBased(const sensor_msgs::CameraInfo& camerainfo,
                     cv::Size compute_size,
                     std::string name
                     );

  virtual bool Process(const cv::Mat rgb,
                       const cv::Mat depth,
                       cv::Mat& marker,
                       cv::Mat& convex_edge,
                       cv::Mat& depthmap,
                       std::map<int,int>& instance2class,
                       bool verbose=false);

protected:

  bool _Process(cv::Mat rgb,
                cv::Mat depth,
                cv::Mat& marker,
                cv::Mat& edge_distance,
                std::map<int,int>& instance2class,
                bool verbose=false);

  virtual void GetEdge(const cv::Mat rgb,
                       const cv::Mat depth,
                       const cv::Mat validmask,
                       cv::Mat& outline_edge,
                       cv::Mat& convex_edge,
                       cv::Mat& surebox_mask,
                       bool verbose) = 0;

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
  virtual void GetEdge(const cv::Mat rgb,
                       const cv::Mat depth,
                       const cv::Mat validmask,
                       cv::Mat& outline_edge,
                       cv::Mat& convex_edge,
                       cv::Mat& surebox_mask,
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
  virtual void GetEdge(const cv::Mat rgb,
                       const cv::Mat depth,
                       const cv::Mat validmask,
                       cv::Mat& outliene_edge,
                       cv::Mat& convex_edge,
                       cv::Mat& surebox_mask,
                       bool verbose);

  ros::Subscriber edge_subscriber_;
  cv::Mat mask_;
  cv::Mat convex_edge_;
};




cv::Mat GetDepthMask(const cv::Mat depth);
cv::Mat GetEdgeMask(const cv::Mat depthmask);

cv::Mat GetGroove(const cv::Mat marker,
                  const cv::Mat depth,
                  const cv::Mat validmask,
                  const cv::Mat vignett,
                  int bg_idx);


#endif
