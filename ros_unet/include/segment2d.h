#ifndef SEGMENT2D_H_
#define SEGMENT2D_H_

#include <opencv2/opencv.hpp>

class Segment2DAbstract {
public:
  Segment2DAbstract(const std::string& name);

  virtual bool Process(const cv::Mat given_rgb,
                       const cv::Mat given_depth,
                       cv::Mat& marker,
                       cv::Mat& convex_edge,
                       std::map<int,int>& instance2class,
                       bool verbose=false) = 0;

  
protected:
  const std::string name_;
};

class Segment2DEdgeBasedAbstract : public Segment2DAbstract{
public:
  Segment2DEdgeBasedAbstract(const std::string& name);

  virtual bool Process(const cv::Mat rgb,
                       const cv::Mat depth,
                       cv::Mat& marker,
                       cv::Mat& convex_edge,
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

class Segment2DEdgeBased : public Segment2DEdgeBasedAbstract {
public:
  Segment2DEdgeBased(const std::string& name);
  void SetEdge(const cv::Mat outline_edge,
               const cv::Mat convex_edge,
               const cv::Mat surebox_mask);

protected:
  virtual void GetEdge(const cv::Mat rgb,
                       const cv::Mat depth,
                       const cv::Mat validmask,
                       cv::Mat& outline_edge,
                       cv::Mat& convex_edge,
                       cv::Mat& surebox_mask,
                       bool verbose);

  cv::Mat outline_edge_;
  cv::Mat convex_edge_;
  cv::Mat surebox_mask_;
};

class Segment2Dthreshold : public Segment2DEdgeBasedAbstract {
public:
  Segment2Dthreshold(const std::string& name, double lap_depth_threshold);
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


cv::Mat GetDepthMask(const cv::Mat depth);
cv::Mat GetEdgeMask(const cv::Mat depthmask);

cv::Mat GetGroove(const cv::Mat marker,
                  const cv::Mat depth,
                  const cv::Mat validmask,
                  const cv::Mat vignett,
                  int bg_idx);

cv::Mat ExpandOutline(const cv::Mat depth, const cv::Mat outline, float fx, float fy);

#endif
