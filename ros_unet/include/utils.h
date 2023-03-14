#ifndef UTILS_H_
#define UTILS_H_

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

template <typename K, typename T>
using EigenMap = std::map<K, T, std::less<K>, Eigen::aligned_allocator<std::pair<const K, T> > >;

template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T> >;
extern std::vector<cv::Scalar> colors;

cv::Mat GetColoredLabel(cv::Mat mask, bool put_text=false);
void HighlightBoundary(const cv::Mat marker, cv::Mat& dst);
cv::Mat GetBoundary(const cv::Mat marker, int w=1);
cv::Mat Overlap(cv::Mat bg, cv::Mat mask, float alpha, bool put_text=false);

uint64_t GetMilliSec();

void ColorizeSegmentation(const std::map<int, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZLNormal> > >& clouds,
                          sensor_msgs::PointCloud2& msg);

void ColorizeSegmentation(const std::map<int, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > >& clouds,
                          sensor_msgs::PointCloud2& msg);

void ColorizeSegmentation(const std::map<int, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZL> > >& clouds,
                          sensor_msgs::PointCloud2& msg);

#endif

