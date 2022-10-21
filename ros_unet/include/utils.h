#ifndef UTILS_H_
#define UTILS_H_

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

template <typename K, typename T>
using EigenMap = std::map<K, T, std::less<K>, Eigen::aligned_allocator<std::pair<const K, T> > >;

template <typename T>
using EigenVector = std::vector<T, Eigen::aligned_allocator<T> >;
extern std::vector<cv::Scalar> colors;

cv::Mat GetColoredLabel(cv::Mat mask, bool put_text=false);
void HighlightBoundary(const cv::Mat marker, cv::Mat& dst);
cv::Mat Overlap(cv::Mat bg, cv::Mat mask);

uint64_t GetMilliSec();


#endif

