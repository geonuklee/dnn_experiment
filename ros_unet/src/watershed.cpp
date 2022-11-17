#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>


#include <opencv2/opencv.hpp> // TODO Remove after debugging.
#include "utils.h"

void DistanceWatershed(const cv::Mat _dist_fromedge, cv::Mat& _marker){
  const float min_d_fromedge = 2.;
  const int IN_QUEUE = -2; // Pixel visited
  const int WSHED = -1;    // Pixel belongs to watershed
  const cv::Size size = _marker.size();
  cv::Mat _expandmap = cv::Mat::zeros(_marker.rows,_marker.cols, CV_32FC1);

  struct Node {
    int* m;
    int* m_parent;
    float* expd; // expand distance
    const float* ed;

    Node(int* _m, int* _m_parent,
         float* _expd, float* _expd_parent,
         const float* _ed)
      :m(_m), m_parent(_m_parent), expd(_expd), ed(_ed){
        m[0] = IN_QUEUE;
        expd[0] = 1. + _expd_parent[0];
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

  // Current pixel in input image
  int* marker = _marker.ptr<int>();
  // Step size to next row in input image
  // ref) https://answers.opencv.org/question/3854/different-step-size-output-for-cvmatstep1/
  const int mstep = int(_marker.step/sizeof(marker[0]));
  const float* edge_distance = _dist_fromedge.ptr<float>();
  float* expand_distance = _expandmap.ptr<float>();
  const int dstep = int(_dist_fromedge.step/sizeof(edge_distance[0]));

  // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
  int i, j;
  for( j = 0; j < size.width; j++ )
    marker[j] = marker[j + mstep*(size.height-1)] = WSHED;

  int n_instance = 0;
  std::priority_queue<Node> q1, q2;
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
      n_instance = std::max(*m, n_instance);
      if( *ed < min_d_fromedge){
        m[0] = WSHED;
        continue;
      }
      if( m[0] != 0)
        continue;
      if(m[-1] > 0)
        q1.push(Node(m, m-1, expd, expd-1, ed));
      else if(m[1] > 0)
        q1.push(Node(m, m+1, expd, expd+1, ed));
      else if(m[-mstep] > 0)
        q1.push(Node(m, m-mstep, expd, expd-dstep, ed));
      else if(m[mstep] > 0)
        q1.push(Node(m, m+mstep, expd, expd+dstep, ed));
    }
  }
  n_instance += 1;

  std::vector<int> remain_expand_areas, edge_distances;
  edge_distances.resize(n_instance, 9999);
  remain_expand_areas.resize(n_instance, 500);
  std::vector<std::map<int, size_t> > boundary_counts;
  boundary_counts.resize(n_instance);

  int iter = 0;
  /*
  cv::VideoWriter writer; 
  int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
  cv::Size sizeFrame(640,480);
  writer.open("modified_watershed.mp4", codec, 15, sizeFrame, true);
  */
  // First step - Expand each instance in limited area
#define ws_push(idx){ \
  if(k.m[idx] == 0) \
    q1.push(Node(k.m+idx, k.m, k.expd+idx, k.expd, k.ed+idx) ); \
}

  while(!q1.empty()){
    Node k = q1.top(); q1.pop();
    *k.m = *k.m_parent;
    int& area = remain_expand_areas[*k.m];
    int& min_ed = edge_distances[*k.m];
    min_ed = std::min(min_ed, (int)*k.ed);
    if(area < 1){
      *k.m = IN_QUEUE;
      q2.push(k);
      continue;
    }
    if( *k.ed < min_d_fromedge){
      boundary_counts[*k.m][WSHED]++;
      *k.m = WSHED;
      continue;
    }
    area--;
    ws_push(-1);
    ws_push(1);
    ws_push(-mstep);
    ws_push(mstep);
    /*
    if( (++iter) % 100 == 0){
      cv::Mat dst = GetColoredLabel(_marker);
      cv::imshow("ModifiedWatershed", dst);
      char c = cv::waitKey(1);
      //writer.write(dst);
    }
    */
  }
#undef ws_push

  // Second step - Expand each instance in limited range
  {
    cv::Mat _fg = _marker > 0;
    const int mode   = cv::RETR_EXTERNAL;
    const int method = cv::CHAIN_APPROX_SIMPLE;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(_fg,contours,hierarchy,mode,method);
    for(i = 0; i < contours.size(); i++){
      const cv::Point& pt = contours.at(i).at(0);
      const int& m = _marker.at<int>(pt.y,pt.x);
      const int thick = 1+2*edge_distances[m];
      //cv::drawContours(_marker, contours, i, m, thick);
      cv::drawContours(_fg, contours, i, 1, thick);
    }
    unsigned char* fg = _fg.ptr<unsigned char>();
    marker = _marker.ptr<int>();
    for(i = 0; i < _fg.total(); i++)
      if(fg[i] < 1)
        marker[i] = WSHED;
  }

// Counts boundary with edges
#define ws_check(idx){ \
  if(k.m[idx]>0){ \
    if(k.m[idx]!=*k.m) \
      boundary_counts[std::max(*k.m,k.m[idx])][std::min(*k.m,k.m[idx])]++;\
  }  \
  else if(k.m[idx] > WSHED) \
    q2.push(Node(k.m+idx, k.m, k.expd+idx, k.expd, k.ed+idx) ); \
  else if(k.m[idx]>IN_QUEUE) \
    boundary_counts[std::max(*k.m,k.m[idx])][std::min(*k.m,k.m[idx])]++;\
}
  while(!q2.empty()){
    Node k = q2.top(); q2.pop();
    *k.m = *k.m_parent;
    if( *k.ed < min_d_fromedge){
      boundary_counts[*k.m][WSHED]++;
      *k.m = WSHED;
      continue;
    }
    ws_check(-1);
    ws_check(1);
    ws_check(mstep);
    ws_check(-mstep);
    /*
    if( (++iter) % 100 == 0){
      cv::Mat dst = GetColoredLabel(_marker);
      cv::imshow("ModifiedWatershed", dst);
      if(cv::waitKey(1) == 'q')
        exit(1);
    }
    */
  }
#undef ws_check

#if 1
  const size_t min_merge_boundary = 50; // [TODO Hard coded param
  {
    std::vector<int*> convert_list;
    std::map<int,std::set<int> > _pairs;
    for(int m0 =0; m0 < n_instance; m0++){
      const auto& counts = boundary_counts[m0];
      for(const auto& it : counts){
        int m1 = it.first;
        if(m1 < 1)
          continue;
        size_t n1 = it.second;
        if( n1 < min_merge_boundary)
          continue;
        _pairs[std::min(m0,m1)].insert(std::max(m0,m1));
      }
    }
    std::vector<int> buf;
    buf.resize(n_instance);
    convert_list.resize(n_instance,nullptr); {
      int* ptr = buf.data();
      for(i=0; i<buf.size(); i++){
        buf[i] = i;
        convert_list[i] = ptr+i;
      }
    }
    for(auto it :_pairs){
      const int& p1 = it.first;
      for(const int& p2 : it.second)
        convert_list[p2] = convert_list[p1];
    }
    for(i=0; i <convert_list.size(); i++){
      int* ptr = convert_list[i];
      if(*ptr==i)
        convert_list[i] = nullptr;
      //std::cout << "cvt " << i << " -> " << *ptr << std::endl;
    }
    //cv::imshow("beforemerge", GetColoredLabel(_marker,true));
    marker = _marker.ptr<int>();
    for( i = 1; i < size.height-1; i++ ) {
      marker += mstep;
      for( j = 1; j < size.width-1; j++ ) {
        int* m = marker + j;
        if(*m < 1)
          continue;
        const int* ptr = convert_list[*m];
        if(!ptr)
          continue;
        *m = *ptr;
      }
    }
    //cv::imshow("aftermerge", GetColoredLabel(_marker,true));
    //cv::waitKey();
  }
#endif

  //writer.release();
  return;
}

namespace cv {
// A node represents a pixel to label
struct WSNode {
  int next;
  int mask_ofs;
  int img_ofs;
};

// Queue for WSNodes
struct WSQueue {
  WSQueue() { first = last = 0; }
  int first, last;
};

static int allocWSNodes( std::vector<WSNode>& storage ) {
  int sz = (int)storage.size();
  int newsz = MAX(128, sz*3/2);
  storage.resize(newsz);
  if( sz == 0 ) {
    storage[0].next = 0;
    sz = 1;
  }
  for( int i = sz; i < newsz-1; i++ )
    storage[i].next = i+1;
  storage[newsz-1].next = 0;
  return sz;
}

}


void ModifiedWatershed(cv::InputArray _src, cv::InputOutputArray _markers){
  using namespace cv;
  // Labels for pixels
  const int IN_QUEUE = -2; // Pixel visited
  const int WSHED = -1; // Pixel belongs to watershed

  // possible bit values = 2^8
  const int NQ = 256;

  Mat src = _src.getMat(), dst = _markers.getMat();
  Size size = src.size();

  // Vector of every created node
  std::vector<WSNode> storage;
  int free_node = 0, node;
  // Priority queue of queues of nodes
  // from high priority (0) to low priority (255)
  WSQueue q[NQ];
  // Non-empty queue with highest priority
  int active_queue;
  int i, j;
  // Color differences
  int db, dg, dr;
  int subs_tab[513];

  // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
  // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

  // Create a new node with offsets mofs and iofs in queue idx
#define ws_push(idx,mofs,iofs)          \
  {                                       \
    if( !free_node )                    \
    free_node = allocWSNodes( storage );\
    node = free_node;                   \
    free_node = storage[free_node].next;\
    storage[node].next = 0;             \
    storage[node].mask_ofs = mofs;      \
    storage[node].img_ofs = iofs;       \
    if( q[idx].last )                   \
    storage[q[idx].last].next=node; \
    else                                \
    q[idx].first = node;            \
    q[idx].last = node;                 \
  }

  // Get next node from queue idx
#define ws_pop(idx,mofs,iofs)           \
  {                                       \
    node = q[idx].first;                \
    q[idx].first = storage[node].next;  \
    if( !storage[node].next )           \
    q[idx].last = 0;                \
    storage[node].next = free_node;     \
    free_node = node;                   \
    mofs = storage[node].mask_ofs;      \
    iofs = storage[node].img_ofs;       \
  }

  // Get highest absolute channel difference in diff
#define c_diff(ptr1,ptr2,diff)           \
  {                                        \
    db = std::abs((ptr1)[0] - (ptr2)[0]);\
    dg = std::abs((ptr1)[1] - (ptr2)[1]);\
    dr = std::abs((ptr1)[2] - (ptr2)[2]);\
    diff = ws_max(db,dg);                \
    diff = ws_max(diff,dr);              \
    CV_Assert( 0 <= diff && diff <= 255 );  \
  }

  CV_Assert( src.type() == CV_8UC3 && dst.type() == CV_32SC1 );
  CV_Assert( src.size() == dst.size() );

  // Current pixel in input image
  const uchar* img = src.ptr();
  // Step size to next row in input image
  int istep = int(src.step/sizeof(img[0]));

  // Current pixel in mask image
  int* mask = dst.ptr<int>();
  // Step size to next row in mask image
  int mstep = int(dst.step / sizeof(mask[0]));

  for( i = 0; i < 256; i++ )
    subs_tab[i] = 0;
  for( i = 256; i <= 512; i++ )
    subs_tab[i] = i - 256;

  // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
  for( j = 0; j < size.width; j++ )
    mask[j] = mask[j + mstep*(size.height-1)] = WSHED;

  // initial phase: put all the neighbor pixels of each marker to the ordered queue -
  // determine the initial boundaries of the basins
  for( i = 1; i < size.height-1; i++ )
  {
    img += istep; mask += mstep;
    mask[0] = mask[size.width-1] = WSHED; // boundary pixels

    for( j = 1; j < size.width-1; j++ )
    {
      int* m = mask + j;
      if( m[0] < 0 ) m[0] = 0;
      if( m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0) )
      {
        // Find smallest difference to adjacent markers
        const uchar* ptr = img + j*3;
        int idx = 256, t;
        if( m[-1] > 0 )
          c_diff( ptr, ptr - 3, idx );
        if( m[1] > 0 )
        {
          c_diff( ptr, ptr + 3, t );
          idx = ws_min( idx, t );
        }
        if( m[-mstep] > 0 )
        {
          c_diff( ptr, ptr - istep, t );
          idx = ws_min( idx, t );
        }
        if( m[mstep] > 0 )
        {
          c_diff( ptr, ptr + istep, t );
          idx = ws_min( idx, t );
        }

        // Add to according queue
        CV_Assert( 0 <= idx && idx <= 255 );
        ws_push( idx, i*mstep + j, i*istep + j*3 );
        m[0] = IN_QUEUE;
      }
    }
  }

  // find the first non-empty queue
  for( i = 0; i < NQ; i++ )
    if( q[i].first )
      break;

  // if there is no markers, exit immediately
  if( i == NQ )
    return;

  active_queue = i;
  img = src.ptr();
  mask = dst.ptr<int>();

  // recursively fill the basins
  for(;;)
  {
    int mofs, iofs;
    int lab = 0, t;
    int* m;
    const uchar* ptr;

    // Get non-empty queue with highest priority
    // Exit condition: empty priority queue
    if( q[active_queue].first == 0 )
    {
      for( i = active_queue+1; i < NQ; i++ )
        if( q[i].first )
          break;
      if( i == NQ )
        break;
      active_queue = i;
    }

    // Get next node
    ws_pop( active_queue, mofs, iofs );

    // Calculate pointer to current pixel in input and marker image
    m = mask + mofs;
    ptr = img + iofs;

    // Check surrounding pixels for labels
    // to determine label for current pixel
    t = m[-1]; // Left
    if( t > 0 ) lab = t;
    t = m[1]; // Right
    if( t > 0 )
    {
      if( lab == 0 ) lab = t;
      else if( t != lab ) lab = WSHED;
    }
    t = m[-mstep]; // Top
    if( t > 0 )
    {
      if( lab == 0 ) lab = t;
      else if( t != lab ) lab = WSHED;
    }
    t = m[mstep]; // Bottom
    if( t > 0 )
    {
      if( lab == 0 ) lab = t;
      else if( t != lab ) lab = WSHED;
    }

    // Set label to current pixel in marker image
    CV_Assert( lab != 0 );
    m[0] = lab;

    if( lab == WSHED )
      continue;

    // Add adjacent, unlabeled pixels to corresponding queue
    if( m[-1] == 0 )
    {
      c_diff( ptr, ptr - 3, t );
      ws_push( t, mofs - 1, iofs - 3 );
      active_queue = ws_min( active_queue, t );
      m[-1] = IN_QUEUE;
    }
    if( m[1] == 0 )
    {
      c_diff( ptr, ptr + 3, t );
      ws_push( t, mofs + 1, iofs + 3 );
      active_queue = ws_min( active_queue, t );
      m[1] = IN_QUEUE;
    }
    if( m[-mstep] == 0 )
    {
      c_diff( ptr, ptr - istep, t );
      ws_push( t, mofs - mstep, iofs - istep );
      active_queue = ws_min( active_queue, t );
      m[-mstep] = IN_QUEUE;
    }
    if( m[mstep] == 0 )
    {
      c_diff( ptr, ptr + istep, t );
      ws_push( t, mofs + mstep, iofs + istep );
      active_queue = ws_min( active_queue, t );
      m[mstep] = IN_QUEUE;
    }
  }
}
