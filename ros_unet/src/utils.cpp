#include "utils.h"

#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

std::vector<cv::Scalar> colors = {
  CV_RGB(0,180,0),
  CV_RGB(0,100,0),
  CV_RGB(255,0,255),
  CV_RGB(100,0,255),
  CV_RGB(100,0,100),
  CV_RGB(0,0,180),
  CV_RGB(0,0,100),
  CV_RGB(255,255,0),
  CV_RGB(100,255,0),
  CV_RGB(100,100,0),
  CV_RGB(100,0,0),
  CV_RGB(0,255,255),
  CV_RGB(0,100,255),
  CV_RGB(0,255,100),
  CV_RGB(0,100,100)
};

uint64_t GetMilliSec(){
  auto t = std::chrono::high_resolution_clock::now();
  auto tn = t.time_since_epoch();
  uint64_t tm = (uint64_t)(tn.count()/1000000.0);
  return tm;
}

cv::Mat Overlap(cv::Mat bg, cv::Mat mask, bool put_text) {
  cv::Mat colored_bg;
  if(bg.channels() > 1)
    colored_bg = bg;
  else
    cv::cvtColor(bg, colored_bg, cv::COLOR_GRAY2BGR);

  cv::Mat colored_mask;
  if(mask.channels() > 1)
    colored_mask = mask;
  else
    colored_mask = GetColoredLabel(mask, put_text);
  HighlightBoundary(mask,colored_mask);
  cv::Mat dst;
  cv::addWeighted(bg, 0.5, colored_mask, 0.5, 0., dst);
  return dst;
}

cv::Mat GetColoredLabel(cv::Mat mask, bool put_text){
  cv::Mat dst = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);
  std::map<int, cv::Point> annotated_lists;

  cv::Mat connected_labels, stats, centroids;
  cv::Mat binary = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);
  for(size_t i = 0; i < mask.rows; i++){
    for(size_t j = 0; j < mask.cols; j++){
      int idx;
      if(mask.type() == CV_8UC1)
        idx = mask.at<unsigned char>(i,j);
      else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
        idx = mask.at<int>(i,j);
      else
        throw "Unexpected type";
      if(idx > 1)
        binary.at<unsigned char>(i,j) = 1;
    }
  }
  cv::connectedComponentsWithStats(binary, connected_labels, stats, centroids, 4);

  for(int i=0; i<stats.rows; i++) {
    int x = centroids.at<double>(i, 0);
    int y = centroids.at<double>(i, 1);
    if(x < 0 or y < 0 or x >= mask.cols or y >= mask.cols)
      continue;

    int idx;
    if(mask.type() == CV_8UC1)
      idx = mask.at<unsigned char>(y,x);
    else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
      idx = mask.at<int>(y,x);

    if(idx > 1 && !annotated_lists.count(idx) ){
      bool overlaped=false;
      cv::Point pt(x,y);
      for(auto it : annotated_lists){
        cv::Point e(pt - it.second);
        if(std::abs(e.x)+std::abs(e.y) < 20){
          overlaped = true;
          break;
        }
      }
      if(!overlaped)
        annotated_lists[idx] = pt;
    }
  }

  for(size_t i = 0; i < mask.rows; i++){
    for(size_t j = 0; j < mask.cols; j++){
      int idx;
      if(mask.type() == CV_8UC1)
        idx = mask.at<unsigned char>(i,j);
      else if(mask.type() == CV_32S) // TODO Unify type of marker map to CV_32S
        idx = mask.at<int>(i,j);
      else
        throw "Unexpected type";
      if(mask.type() == CV_8UC1 && idx == 0)
        continue;
      else if(mask.type() == CV_32S && idx < 0)
        continue;

      cv::Scalar bgr;
      if( idx == 0)
        bgr = CV_RGB(100,100,100);
      //else if (idx == 1)
      //  bgr = CV_RGB(255,255,255);
      else
        bgr = colors.at( idx % colors.size() );

      dst.at<cv::Vec3b>(i,j)[0] = bgr[0];
      dst.at<cv::Vec3b>(i,j)[1] = bgr[1];
      dst.at<cv::Vec3b>(i,j)[2] = bgr[2];

      if(idx > 1 && !annotated_lists.count(idx) ){
        bool overlaped=false;
        cv::Point pt(j,i+10);
        for(auto it : annotated_lists){
          cv::Point e(pt - it.second);
          if(std::abs(e.x)+std::abs(e.y) < 20){
            overlaped = true;
            break;
          }
        }
        if(!overlaped)
          annotated_lists[idx] = pt;
      }
    }
  }

  if(put_text){
    for(auto it : annotated_lists){
      //cv::rectangle(dst, it.second+cv::Point(0,-10), it.second+cv::Point(20,0), CV_RGB(255,255,255), -1);
      const auto& c0 = colors.at( it.first % colors.size() );
      //const auto color = c0;
      const auto color = (c0[0]+c0[1]+c0[2] > 255*2) ? CV_RGB(0,0,0) : CV_RGB(255,255,255);
      cv::putText(dst, std::to_string(it.first), it.second, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
  }
  return dst;
}

void HighlightBoundary(const cv::Mat marker, cv::Mat& dst){
  const int w = 1;
  for(int r0 = 0; r0 < marker.rows; r0++){
    for(int c0 = 0; c0 < marker.cols; c0++){
      const int& i0 = marker.at<int>(r0,c0);
      bool b = false;
      for(int r1 = std::max(r0-w,0); r1 < std::min(r0+w,marker.rows); r1++){
        for(int c1 = std::max(c0-w,0); c1 < std::min(c0+w,marker.cols); c1++){
          const int& i1 = marker.at<int>(r1,c1);
          b = i0 != i1;
          if(b)
            break;
        }
        if(b)
          break;
      }
      if(!b)
        continue;
      dst.at<cv::Vec3b>(r0,c0)[0] = 255;
      dst.at<cv::Vec3b>(r0,c0)[1] = 255;
      dst.at<cv::Vec3b>(r0,c0)[2] = 255;
    }
  }
  return;
}

cv::Mat GetBoundary(const cv::Mat marker, int w){
  cv::Mat boundarymap = cv::Mat::zeros(marker.rows,marker.cols, CV_8UC1);
  for(int r0 = 0; r0 < marker.rows; r0++){
    for(int c0 = 0; c0 < marker.cols; c0++){
      const int& i0 = marker.at<int>(r0,c0);
      bool b = false;
      for(int r1 = std::max(r0-w,0); r1 < std::min(r0+w,marker.rows); r1++){
        for(int c1 = std::max(c0-w,0); c1 < std::min(c0+w,marker.cols); c1++){
          const int& i1 = marker.at<int>(r1,c1);
          b = i0 != i1;
          if(b)
            break;
        }
        if(b)
          break;
      }
      if(!b)
        continue;
      boundarymap.at<unsigned char>(r0,c0) = true;
    }
  }
  return boundarymap;
}

