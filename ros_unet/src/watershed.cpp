#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>


#include <opencv2/opencv.hpp> // TODO Remove after debugging.
#include "utils.h"

void DistanceWatershed(const cv::Mat _dist_fromedge, cv::Mat& _marker){
  const int IN_QUEUE = -2; // Pixel visited
  const int WSHED = -1;    // Pixel belongs to watershed
  const cv::Size size = _marker.size();
  cv::Mat _expandmap = cv::Mat::zeros(_marker.rows,_marker.cols, CV_32FC1);
  struct Node {
    int* m;
    int* m_parent;
    float* expd;
    float* expd_parent;
    const float* ed;

    Node(int* _m, int* _m_parent,
         float* _expd, float* _expd_parent,
         const float* _ed)
      :m(_m), m_parent(_m_parent), expd(_expd), expd_parent(_expd_parent), ed(_ed){
        m[0] = IN_QUEUE;
        expd[0] = 1. + expd_parent[0];
    }

    bool operator < (const Node& other) const{
      int a = static_cast<int>(*ed);
      int b = static_cast<int>(*other.ed);
      if( a < b )
        return true;
      else if( a > b)
        return false;
      else if(expd[0] > other.expd[0])
        return true;
      return false;
    }
  };
  //std::vector<bool> visited;
  //visited.resize(_marker.rows*_marker.cols);
  std::priority_queue<Node> q;

  // Current pixel in input image
  int* marker = (int*)_marker.ptr();
  // Step size to next row in input image
  // ref) https://answers.opencv.org/question/3854/different-step-size-output-for-cvmatstep1/
  const int mstep = int(_marker.step/sizeof(marker[0]));
  const float* edge_distance = (float*) _dist_fromedge.ptr();
  float* expand_distance = (float*) _expandmap.ptr();
  const int dstep = int(_dist_fromedge.step/sizeof(edge_distance[0]));

  // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
  int i, j;
  for( j = 0; j < size.width; j++ )
    marker[j] = marker[j + mstep*(size.height-1)] = WSHED;
  for( i = 1; i < size.height-1; i++ ) {
    marker += mstep;
    edge_distance += dstep;
    expand_distance += dstep;
    marker[0] = marker[size.width-1] = WSHED; // boundary pixels

    // initial phase: put all the neighbor pixels of each marker to the priority queue -
    // determine the initial boundaries of the basins
    for( j = 1; j < size.width-1; j++ ) {
      int* m = marker + j;
      float* expd     = expand_distance + j;
      const float* ed = edge_distance   + j;
      if( *ed < 1.){
        m[0] = WSHED;
        continue;
      }
      if( m[0] != 0)
        continue;
      if(m[-1] > 0)
        q.push(Node(m, m-1, expd, expd-1, ed));
      else if(m[1] > 0)
        q.push(Node(m, m+1, expd, expd+1, ed));
      else if(m[-mstep] > 0)
        q.push(Node(m, m-mstep, expd, expd-dstep, ed));
      else if(m[mstep] > 0)
        q.push(Node(m, m+mstep, expd, expd+dstep, ed));
    }
  }

  /*
  int iter = 0;
  cv::VideoWriter writer; 
  int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
  cv::Size sizeFrame(640,480);
  writer.open("modified_watershed.mp4", codec, 15, sizeFrame, true);
  */

  while(!q.empty()){
    Node k = q.top(); q.pop();
    *k.m = *k.m_parent;
    if(k.m[-1] == 0)
      q.push(Node(k.m-1, k.m, k.expd-1, k.expd, k.ed-1) );
    if(k.m[1] == 0)
      q.push(Node(k.m+1, k.m, k.expd+1, k.expd, k.ed+1) );
    if(k.m[-mstep] == 0)
      q.push(Node(k.m-mstep, k.m, k.expd-dstep, k.expd, k.ed-dstep) );
    if(k.m[mstep] == 0)
      q.push(Node(k.m+mstep, k.m, k.expd+dstep, k.expd, k.ed+dstep) );
    /*
    if( (++iter) % 100 == 0){
      cv::Mat dst = GetColoredLabel(_marker);
      writer.write(dst);
      cv::imshow("ModifiedWatershed", dst);
      char c = cv::waitKey(1);
    }
    */
  }
  /*
  writer.release();
  */
  return;
}

