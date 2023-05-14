#include "segment2d.h"
#include "utils.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>

int Convert(const std::map<int,int>& convert_lists,
             const std::set<int>& leaf_seeds,
             const cv::Mat& filtered_edge, 
             const cv::Mat& depthmask,
             cv::Mat& seed ){
  int bg_idx = std::max<int>(100, (*leaf_seeds.rbegin()) + 1);
  int* ptr = (int*) seed.data;
  const int n = seed.rows*seed.cols;

  for(int r = 0; r < n; r++){
      int& idx = ptr[r];
      //if(filtered_edge.data[r] > 0)
      //  idx = 1;
      if(depthmask.data[r] < 1)
        idx = bg_idx;
      else if(convert_lists.count(idx) )
        idx = convert_lists.at(idx);
      else if(!leaf_seeds.count(idx) )
        idx = 0;
  }

  return bg_idx;
}

void DrawOutline(cv::Mat edges,
                 cv::Mat depthmask,
                 cv::Mat& outlined_rgb){
  for(int r = 0; r < edges.rows; r++){
    for(int c = 0; c < edges.cols; c++){
      bool need_outline = false;
      if(edges.at<unsigned char>(r,c) > 0)
        need_outline = true;
      else if(depthmask.at<unsigned char>(r,c) < 1)
        need_outline = true;
      if(!need_outline)
        continue;
      auto& vec3b = outlined_rgb.at<cv::Vec3b>(r,c);
      vec3b[0] = vec3b[1] = vec3b[2] = 0;
    }
  }
  return;
}

void GetThickClosedEdge(cv::Mat marker, cv::Mat& thick_closed_edge){
  thick_closed_edge = cv::Mat::zeros(marker.rows, marker.cols, CV_8UC1);
  for(int r = 0; r < marker.rows; r++){
    for(int c = 0; c < marker.cols; c++){
      int& idx = marker.at<int>(r,c);
      if(r == 0 || r ==  marker.rows-1)
          thick_closed_edge.at<unsigned char>(r,c) = 0;
      else if(c == 0 || c ==  marker.cols-1)
          thick_closed_edge.at<unsigned char>(r,c) = 0;
      else if(idx  < 2) // remap -1,0,1 to 1
          thick_closed_edge.at<unsigned char>(r,c) = 1;
      else
        thick_closed_edge.at<unsigned char>(r,c) = 0;
    }
  }
  cv::Mat ones = cv::Mat::ones(7,7,CV_8UC1);
  cv::dilate(thick_closed_edge, thick_closed_edge, ones, cv::Point(-1,-1),1,cv::BORDER_DEFAULT);
  cv::threshold(thick_closed_edge, thick_closed_edge, 0, 1, cv::THRESH_BINARY);
  return;
}


cv::Mat GetGroove(const cv::Mat marker,
                  const cv::Mat depth,
                  const cv::Mat validmask,
                  const cv::Mat vignett32,
                  int bg_idx) {

  cv::Mat groove = cv::Mat::zeros(marker.rows, marker.cols, CV_8UC1);
  const int w = 2;
  const int& rows = groove.rows;
  const int& cols = groove.cols;
  for(int r0 = 0; r0 < rows; r0++){
    for(int c0 = 0; c0 < cols; c0++){
      const int& i0 = marker.at<int>(r0,c0);
      bool b = false;
      for(int r1 = r0-w; r1 < r0+w; r1++){
        for(int c1 = c0-w; c1 < c0+w; c1++){
          if(r1 < 0 || r1 >= rows || c1 < 0 || c1 >= cols)
            continue;
          const int& i1 = marker.at<int>(r1,c1);
          b = i0 != i1;
          if(b)
            break;
        }
        if(b)
          break;
      }
      groove.at<unsigned char>(r0,c0) = !b;
    }
  }
  return groove;
}

void GetInside(cv::Mat marker, int bg_idx, cv::Mat& edges){
  edges = cv::Mat::zeros(marker.rows, marker.cols, CV_8UC1);
  for(int r = 0; r < marker.rows; r++){
    for(int c = 0; c < marker.cols; c++){
      const int& v = marker.at<int>(r,c);
      bool inside = true;
      if(v>1 && v < bg_idx){
        if(c+1<marker.cols)
          inside &= marker.at<int>(r,c+1) == v;
        if(r+1<marker.rows)
          inside &= marker.at<int>(r,c+1) == v;
        if(c+1<marker.cols && r+1<marker.rows)
          inside &= marker.at<int>(r+1,c+1) == v;
      }
      else
        inside = false;
      edges.at<unsigned char>(r,c) = inside ? 1 : 0;
    }
  }
  return;
}

void GetVornoi(const cv::Mat marker, const cv::Mat& closed_edge, int bg_idx,
               cv::Mat& voronoi){
  std::map<int, size_t> marker_size;
  std::map<int, std::map<int, size_t> > voronoi2marker_stats;
  for(int r = 0; r < marker.rows; r++){
    for(int c = 0; c < marker.cols; c++){
      const int& marker_idx = marker.at<int>(r,c);
      int& vornoi_idx = voronoi.at<int>(r,c);
      marker_size[marker_idx]++;
      voronoi2marker_stats[vornoi_idx][marker_idx]++;
    }
  }

  std::map<int, int> voronoi2marker;
  for(auto it : voronoi2marker_stats){
    int champ = -1;
    size_t n = 0;
    for(auto it2 : it.second){
      if(it2.second > n){
        champ = it2.first;
        n = it2.second;
      }
    }
    double r =  (double) n / (double)marker_size.at(champ);
    if(r > 0.8)
      voronoi2marker[it.first] = champ;
  }

  for(int r = 0; r < marker.rows; r++){
    for(int c = 0; c < marker.cols; c++){
      const int& marker_idx = marker.at<int>(r,c);
      int& voronoi_idx = voronoi.at<int>(r,c);
      if(voronoi2marker.count(voronoi_idx))
        voronoi_idx = voronoi2marker.at(voronoi_idx);
      else
        voronoi_idx = 0;
      if(voronoi_idx == bg_idx)
        voronoi_idx = 0;
    }
  }

  return;
}

void Revert(int bg_idx, cv::Mat& marker){
  for(int r = 0; r < marker.rows; r++){
    for(int c = 0; c < marker.cols; c++){
      int& idx = marker.at<int>(r,c);
      if(idx == bg_idx) // remove bg marker.
        idx = 0;
      else if(idx < 2) // remove edge(1, -1) marker.
        idx = 0;
    }
  }
  return;
}

cv::Mat EstimateNormal(cv::Mat K, cv::Mat depth){
  const float fx = K.at<float>(0,0);
  const float fy = K.at<float>(1,1);
  const int hw = 3;
  const float w = hw*2.;
  cv::Mat cosmap = cv::Mat::zeros(depth.rows, depth.cols, CV_32F);
  for(int r = 0; r < cosmap.rows; r++){
    if(r-hw < 0)
      continue;
    if(r+hw >= depth.rows)
      continue;
    for(int c = 0; c < cosmap.cols; c++){
      if(c-hw < 0)
        continue;
      if(c+hw >= depth.cols)
        continue;
      const float& z     = depth.at<float>(r,c);
      if( z < 0.001)
        continue;
      const float& z_u0  = depth.at<float>(r, c-hw);
      const float& z_u1  = depth.at<float>(r, c+hw);
      const float& z_v0  = depth.at<float>(r-hw, c);
      const float& z_v1  = depth.at<float>(r+hw, c);
      float dzdu = (z_u1 - z_u0)/w;
      float dzdv = (z_v1 - z_v0)/w;
      float dudx = fx/z;
      float dvdy = fy/z;
      float dzdx = dzdu*dudx;
      float dzdy = dzdv*dvdy;
      float cth = 1./std::sqrt(1.+dzdx*dzdx+dzdy*dzdy);
      cosmap.at<float>(r,c) = cth;
    }
  }
  return cosmap;
}

Segment2DAbstract::Segment2DAbstract(const std::string& name)
: name_(name)
{

}

Segment2DEdgeBasedAbstract::Segment2DEdgeBasedAbstract(const std::string& name)
  : Segment2DAbstract(name)
{

}

cv::Mat GetDepthMask(const cv::Mat depth) {
  cv::Mat depthmask;
#if 1
  // TODO Remove hard code considering std_2021.08. experiment.
  depthmask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  for(int r = 0; r < depth.rows; r++){
    for(int c = 0; c < depth.cols; c++){
      const float& d = depth.at<float>(r, c);
      if(d > 0.01 && d < 999.) // TODO max depth?
          depthmask.at<unsigned char>(r,c) = 255;
    }
  }
#else
  cv::threshold(depth, depthmask, 0.1, 255, cv::THRESH_BINARY);
#endif

  const cv::Mat erode_kernel = cv::Mat::ones(5, 5, CV_32F);
  //cv::erode(depthmask, depthmask, erode_kernel, cv::Point(-1,-1), 1);
  cv::dilate(depthmask, depthmask, erode_kernel, cv::Point(-1,-1), 1);
  cv::threshold(depthmask, depthmask, 200, 1, cv::THRESH_BINARY);

  depthmask.convertTo(depthmask, CV_8UC1);
  return depthmask;
}

cv::Mat GetDiscontinuousDepthEdge(const cv::Mat& depth,
                               float threshold_depth){
  const int& rows = depth.rows;
  const int& cols = depth.cols;
  const int size = rows*cols;
  const int hk = 1;
  const std::vector<int> deltas = {-hk, hk};
  cv::Mat output = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  for (int r0=hk; r0<rows-hk; r0++) {
    for (int c0=hk; c0<cols-hk; c0++) {
      const float& d_cp = depth.at<float>(r0,c0);
      for(const int& r_delta : deltas){
        const int r = r0 + r_delta;
        if( output.at<unsigned char>(r0,c0) )
          break;
        for(const int& c_delta : deltas){
          const int c = c0 + c_delta;
          if( output.at<unsigned char>(r0,c0) )
            break;
          const float& d = depth.at<float>(r,c);
#if 0
          bool c1 = d < 0.001;
          bool c2 = d_cp < 0.001;
          if(c1 || c2)
            continue;
          bool c3 = std::abs(d-d_cp) > threshold_depth;
          if(c3){
            output.at<unsigned char>(r0,c0) = true;
            break;
          } // if abs(d-d_cp) > threshold
#else
          bool c1 = d < 0.001;
          bool c2 = d_cp < 0.001;
          bool c3 = std::abs(d-d_cp) > threshold_depth;
          if(c1 || c2 || c3){
            output.at<unsigned char>(r0,c0) = true;
            break;
          } // if abs(d-d_cp) > threshold
#endif
        } // for c
      } //for r
    } // for c0
  } // for r0
  return output;
}

cv::Mat GetForegroundMask(const cv::Mat depth){
  const float threshold_depth = .2;
  cv::Mat dd_edge= GetDiscontinuousDepthEdge(depth, threshold_depth);
  cv::imshow("dd_edge", dd_edge*255);
  cv::waitKey(1);
  return dd_edge;
}

cv::Mat GetValidMask(const cv::Mat depthmask) {
    const int mode   = cv::RETR_EXTERNAL; // RETR_CCOMP -> RETR_EXTERNAL
    const int method = cv::CHAIN_APPROX_SIMPLE;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(depthmask, contours, hierarchy, mode, method);
    int thickness = 9; // Greater than kernel of lap.
    cv::Mat edge_mask = cv::Mat::zeros(depthmask.rows, depthmask.cols, CV_8UC1);
    for(int i = 0; i < contours.size(); i++)
      cv::drawContours(edge_mask, contours, i, 1, -1);
    for(int i = 0; i < contours.size(); i++)
      cv::drawContours(edge_mask, contours, i, 0, thickness);
    return edge_mask;
}

bool Segment2DEdgeBasedAbstract::Process(const cv::Mat rgb,
                                 const cv::Mat depth,
                                 cv::Mat& marker,
                                 cv::Mat& convex_edge,
                                 std::map<int,int>& instance2class,
                                 bool verbose){
  return _Process(rgb, depth, marker, convex_edge, instance2class, verbose);
}

std::set<int> GetUnique(const cv::Mat marker){
  std::set<int> unique;
  int n = marker.total();
  const int* ptr = marker.ptr<int>();
  for(int i=0; i < n; i++)
    unique.insert(ptr[i]);
  return unique;
}

void PrintUnique(const cv::Mat marker){
  std::set<int> unique = GetUnique(marker);
  for(const auto& i : unique)
    std::cout << i << ", ";
  std::cout << std::endl;
  return;
}

cv::Mat Unify(cv::Mat _marker){
  cv::Mat _output = cv::Mat::zeros(_marker.rows, _marker.cols, CV_32SC1);
  int* marker = _marker.ptr<int>();
  int* output = _output.ptr<int>();
  std::map<int,int> cvt_map;
  cvt_map[0] = 0;
  int n = _marker.total();
  for(int i=0; i<n; i++){
    const int& m0 = marker[i];
    if(!cvt_map.count(m0))
      cvt_map[m0] = cvt_map.size();
    output[i] = cvt_map[m0];
  }
  return _output;
}

bool IsTooSmallSeed(const cv::RotatedRect& obb, const int& rows, const int& cols){
  // 화면 구석 너무 작은 seed 쪼가리 때문에 oversegment 나는것 방지.
  const cv::Point2f& cp = obb.center;
  const float offset = 10.;
  if(cp.x < offset)
    return true;
  else if(cp.x > cols - offset)
    return true;
  else if(cp.y < offset)
    return true;
  else if(cp.y > rows - offset)
    return true;
  float wh = std::max(obb.size.width, obb.size.height);
  return wh < 20.;
}

bool Segment2DEdgeBasedAbstract::_Process(cv::Mat rgb,
                        cv::Mat depth,
                        cv::Mat& marker,
                        cv::Mat& convex_edge,
                        std::map<int,int>& instance2class,
                        bool verbose){
  cv::Mat depthmask = GetDepthMask(depth);
  cv::Mat outline_edge, validmask;
  GetEdge(rgb, depth, outline_edge, convex_edge, validmask, verbose);

  if(outline_edge.empty())
    return false;

  cv::Mat divided;
  cv::Mat dist_fromoutline; {
#if 1
    cv::distanceTransform(outline_edge<1, dist_fromoutline, cv::DIST_L2, cv::DIST_MASK_PRECISE);
#else
    cv::bitwise_and(depthmask, ~outline_edge, divided);
    cv::bitwise_and(vignett8U_, divided, divided);
    if(divided.type()!=CV_8UC1)
      divided.convertTo(divided, CV_8UC1); // distanceTransform asks CV_8UC1 input.
    cv::distanceTransform(divided, dist_transform, cv::DIST_L2, cv::DIST_MASK_3);
#endif
    for(int r=0; r<dist_fromoutline.rows; r++){
      for(int c=0; c<dist_fromoutline.cols; c++){
        if(!validmask.at<unsigned char>(r,c))
          dist_fromoutline.at<float>(r,c) = 0.;
      }
    }
  }
  cv::Mat seedmap = cv::Mat::zeros(depth.rows, depth.cols, CV_32SC1);
  cv::Mat seed_contours;
  std::set<int> leaf_nodes;
  int bg_idx = -1;
  const int mode   = cv::RETR_TREE; // RETR_CCOMP -> RETR_EXTERNAL
  const int method = cv::CHAIN_APPROX_SIMPLE;
  double dth = 4.; 
  //if(verbose)
  //  dth = 8.; // For readability of figure
  int n = 100./dth; // max level should be limitted.
  //int n = 1;
  float min_width = 10.;
  std::map<int,std::set<int> > seed_childs;
  std::map<int,int> seed_parents;
  std::map<int,cv::RotatedRect> seed_obbs;
  std::map<int,float> seed_areas;
  std::map<int,int> seed_levels;
  std::map<int,int> seed_dists;
    
  std::vector<std::vector<cv::Point> > vis_contours, vis_invalid_contours;
  {
    int idx = 1; // Assign 1 for edge. Ref) utils.cpp, GetColoredLabel
    for(int lv = 0; lv < n; lv++){
      int th_distance = dth*(float)lv;
      cv::Mat local_seed;
      cv::threshold(dist_fromoutline, local_seed, th_distance, 255, cv::THRESH_BINARY);
      local_seed.convertTo(local_seed, CV_8UC1); // findContour support only CV_8UC1

      std::vector<std::vector<cv::Point> > contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(local_seed, contours, hierarchy, mode, method);
      int n_insertion=0;
      for(size_t j=0; j < contours.size(); j++){
        const std::vector<cv::Point>& cnt = contours.at(j);
        const cv::Vec4i& vec = hierarchy.at(j);
        if(vec[3] > -1)
          continue;
        const int& x = cnt.at(0).x;
        const int& y = cnt.at(0).y;
        const int exist_idx = seedmap.at<int>(y,x);
        const cv::RotatedRect ar = cv::minAreaRect(cnt);
        if(std::min(ar.size.width,ar.size.height) < min_width){
          if(verbose)
            vis_invalid_contours.push_back(cnt);
          continue;
        }
        idx += 1;
        assert(exist_idx!=idx);
        n_insertion++;
        seed_childs[exist_idx].insert(idx);
        seed_parents[idx] = exist_idx;
        seed_obbs[idx] = ar;
        seed_areas[idx] = cv::contourArea(cnt);
        seed_levels[idx] = lv;
        seed_dists[idx] = th_distance;
        std::vector<std::vector<cv::Point> > cnts = { cnt, };
        cv::drawContours(seedmap, cnts, 0, idx,-1);
        seed_childs[idx];
        if(verbose)
          vis_contours.push_back(cnt);
      }
      if(n_insertion < 1)
        break;
    }
  }
  seed_contours = seedmap.clone();
  for(const auto& it : seed_childs){
    const int& parent = it.first;
    const std::set<int>& childs = it.second;
    if(parent == 0){
      for(const int& child : childs)
        if(seed_childs.at(child).empty())
          leaf_nodes.insert(child);
    }
    //else if(childs.size() == 1){ // No sibling.
    //  int single_child = *childs.begin();
    //  // When parent has no sibling, the single child is the leaf.
    //  if(seed_childs.at(single_child).empty())
    //    leaf_nodes.insert(single_child);
    //}
    else if( childs.empty() )
      leaf_nodes.insert(parent);
  }
  std::map<int, int> convert_lists;
  // The pairs of highest and lowest contours for each instance.
  std::map<int,int> lowest2highest;
  // The below loop updates convert_lists
  for(const int& idx : leaf_nodes){
    if(convert_lists.count(idx))
      continue;
    std::set<int> contours_under_pole;
    std::priority_queue<int> q1;
    q1.push(idx);
    while(!q1.empty()){
      // Descent from pole to lower contour
      const int keyidx = q1.top();
      q1.pop();
      int parent = seed_parents.at(keyidx);
      std::set<int> siblings_of_key;
      if(seed_childs.count(parent)){
        siblings_of_key = seed_childs.at(parent);
      }
      contours_under_pole.insert(keyidx);
      if(parent==0){
        continue; // No q1.push(parent). Stop descendent.
      }
      else if(siblings_of_key.size() == 1){
        // 자식노드 숫자 count해서 내려가기전,..
        //const cv::RotatedRect& keyidx_obb = seed_obbs.at(keyidx);
        //const cv::RotatedRect& parent_obb = seed_obbs.at(parent);
        //const int& lv = seed_levels.at(keyidx);
        //if(lv > 0 &&  std::min(keyidx_obb.size.width,keyidx_obb.size.height) > 50) {
        //  const float expectation =(keyidx_obb.size.width+2.*dth)*(keyidx_obb.size.height+2.*dth);
        //  const float parent_area = parent_obb.size.width*parent_obb.size.height;
        //  const float err_ratio = std::abs(expectation-parent_area)/parent_area;
        //  if(err_ratio > 0.5)
        //    continue; // No q1.push(parent). Stop descendent.
        //}
        q1.push(parent); // Keep descent to a lower contour.
      }
      else if(siblings_of_key.size()==2) {
#if 1
        continue;
#else
        float sum = 0;
        for(const int& sibling : siblings_of_key)
          sum += seed_areas.at(sibling);
        float parent_area = seed_areas.at(parent);
        const cv::RotatedRect& parent_obb = seed_obbs.at(parent);
        const float ratio = (parent_obb.size.width-2.*dth)*(parent_obb.size.width-2.*dth)/
          (parent_obb.size.width*parent_obb.size.width);
        const float expected_area_sum = ratio*parent_area;
        if(sum > .8*expected_area_sum)
          continue; // No q1.push(parent). Stop descendent.
        else{
          q1.push(parent); // Keep descent to a lower contour.
          // sibling > 1 이지만, 하나로 묶어야 하는 경우 <<<
          for(const int& sibling : siblings_of_key){
            if(sibling == keyidx)
              continue;
            // Get all upper contours of sibling.
            std::queue<int> opened;
            opened.push(sibling);
            while(!opened.empty()){
              int lkeyidx = opened.front();
              opened.pop();
              contours_under_pole.insert(lkeyidx);
              for(const int& child : seed_childs.at(lkeyidx) )
                opened.push(child);
            }
          }
        }
#endif
      }
    }
    int lowest_contour = *contours_under_pole.begin(); // lowest boundary = max(contours_under_pole)
    int highest_contour = *contours_under_pole.rbegin(); // pole = min(contours_under_pole)
    const cv::RotatedRect& lowest_obb = seed_obbs.at(lowest_contour);
    if(IsTooSmallSeed(lowest_obb,seedmap.rows,seedmap.cols)){
      for(const int& i : contours_under_pole)
        convert_lists[i] = 0;
    }
    else{
      lowest2highest[lowest_contour] = highest_contour;
      for(const int& i : contours_under_pole){
        if(i != lowest_contour)
          convert_lists[i] = lowest_contour;
      }
    }
  } // compute leaf_seeds

  if(!lowest2highest.empty()) {
    // Convert elements of seed from exist_idx to convert_lists
    bg_idx = Convert(convert_lists, leaf_nodes,
                     outline_edge, depthmask,
                     seedmap);
  }
  else
    bg_idx = 2;

  seedmap = cv::max(seedmap,0);
  marker = Unify(seedmap);

  cv::Mat vis_arealimitedflood, vis_rangelimitedflood, vis_onedgesflood;
  cv::Mat vis_heightmap, vis_seed;
  if(verbose){
    vis_arealimitedflood  = cv::Mat::zeros(depth.rows,depth.cols, CV_8UC3);
    vis_rangelimitedflood = cv::Mat::zeros(depth.rows,depth.cols, CV_8UC3);
    vis_onedgesflood      = cv::Mat::zeros(depth.rows,depth.cols, CV_8UC3);
    cv::normalize(-dist_fromoutline, vis_heightmap, 255, 0, cv::NORM_MINMAX, CV_8UC1);
    cv::cvtColor(vis_heightmap,vis_heightmap, CV_GRAY2BGR);
    for(int r=0; r<vis_heightmap.rows; r++){
      for(int c=0; c<vis_heightmap.cols; c++){
        if(outline_edge.at<unsigned char>(r,c) <1)
          continue;
        vis_heightmap.at<cv::Vec3b>(r,c)[0] 
          = vis_heightmap.at<cv::Vec3b>(r,c)[1] = 0;
      }
    }
    cv::Mat fg = marker > 0;
    const int mode   = cv::RETR_EXTERNAL;
    const int method = cv::CHAIN_APPROX_SIMPLE;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(fg,contours,hierarchy,mode,method);
    for(int i = 0; i < contours.size(); i++){
      const std::vector<cv::Point>& contour = contours.at(i);
      float min_ed = 99999.;
      for(const cv::Point& pt :  contours.at(i) ){
        const float& ed = dist_fromoutline.at<float>(pt.y,pt.x);
        min_ed = std::min(min_ed, ed);
      }
      const int thick = 1+2*min_ed;
      cv::drawContours(fg, contours, i, 1, thick);
    }

    vis_seed = cv::Mat::zeros(depth.rows,depth.cols, CV_8UC3);
    cv::Mat weights = cv::Mat::zeros(depth.rows,depth.cols, CV_32F);
    for(int i = 0; i < vis_contours.size(); i++){
      const std::vector<cv::Point>& contour = vis_contours.at(i);
      const auto pt = contour.at(0);
      float w = float(vis_heightmap.at<cv::Vec3b>(pt.y, pt.x)[0]) / 255.;
      w = std::min<float>(1.,w);
      w = std::max<float>(0.,w);
      cv::drawContours(weights, vis_contours, i, w, -1);
    }

    for(int r=0; r<marker.rows; r++){
      for(int c=0; c<marker.cols; c++){
        auto& cs = vis_seed.at<cv::Vec3b>(r,c);
        auto& ca = vis_arealimitedflood.at<cv::Vec3b>(r,c);
        auto& w = weights.at<float>(r,c);
        const int& m = marker.at<int>(r,c);
        if(m > 0){
          cs[0] = w * colors.at(m%colors.size())[0];
          cs[1] = w * colors.at(m%colors.size())[1];
          cs[2] = w * colors.at(m%colors.size())[2];
          ca[0] = .5 * colors.at(m%colors.size())[0];
          ca[1] = .5 * colors.at(m%colors.size())[1];
          ca[2] = .5 * colors.at(m%colors.size())[2];
          //ca[0] = ca[1] = ca[2] = 0;
          continue;
        }
        else
          ca[0] = ca[1] = ca[2] = 255;
        const float& ed = dist_fromoutline.at<float>(r,c);
        if(ed > 0){
          if(fg.at<uchar>(r,c) > 0)
            cs[0] = cs[1] = cs[2] = 100;
          else
            cs[0] = cs[1] = cs[2] = 255;
          continue;
        }
        else{
          // ca[0] = ca[1] = 0; ca[2] = 255; // Draw edges as red later.
          vis_seed.at<cv::Vec3b>(r,c)[2] = 255;
          vis_seed.at<cv::Vec3b>(r,c)[0] = vis_seed.at<cv::Vec3b>(r,c)[1] = 0;
        }
      }
    }
    for(int i = 0; i < vis_contours.size(); i++){
      const std::vector<cv::Point>& contour = vis_contours.at(i);
      const auto pt = contour.at(0);
      //uchar w = 255. * weights.at<float>(pt.y,pt.x);
      uchar w = 200;
      cv::drawContours(vis_seed, vis_contours, i, CV_RGB(w,w,w), 1);
    }
    for(int i = 0; i < vis_invalid_contours.size(); i++){
      cv::drawContours(vis_seed, vis_invalid_contours, i, CV_RGB(0,0,0), 1);
    }

  } // if(verbose)
  DistanceWatershed(dist_fromoutline, marker,
                    vis_arealimitedflood,
                    vis_rangelimitedflood,
                    vis_onedgesflood);
  marker = cv::max(marker, 0);

  if(verbose){
    {
      cv::Mat fg = seedmap > 0;
      const int mode   = cv::RETR_EXTERNAL;
      const int method = cv::CHAIN_APPROX_SIMPLE;
      std::vector<std::vector<cv::Point> > contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(fg,contours,hierarchy,mode,method);
      for(int i = 0; i < contours.size(); i++)
        cv::drawContours(vis_arealimitedflood, contours, i, CV_RGB(0,0,0), 2);
      for(int r=0; r<marker.rows; r++){
        for(int c=0; c<marker.cols; c++){
          auto& ca = vis_arealimitedflood.at<cv::Vec3b>(r,c);
          if(outline_edge.at<uchar>(r,c) < 1)
            continue;
          ca[0] = ca[1] = 0; ca[2] = 255;
        }
      }
    }

    //cv::imshow(name_+"marker", GetColoredLabel(marker,true) );
    //cv::imshow(name_+"outline", outline_edge*255);
    //cv::imshow(name_+"seed contour", GetColoredLabel(seed_contours));
    //cv::imshow(name_+"seed", Overlap(rgb,seedmap) );
    //cv::Mat dst_marker = Overlap(rgb,marker);
    //cv::imshow(name_+"final_marker", dst_marker );
    {
      cv::Mat dst = rgb.clone();
      for(int r=0; r<dst.rows; r++){
        for(int c=0; c<dst.cols; c++){
          auto& cd = dst.at<cv::Vec3b>(r,c);
          if(outline_edge.at<uchar>(r,c) < 1)
            continue;
          cd[0] = cd[1] = 0; cd[2] = 255;
        }
      }
      cv::imshow("vis_rgb",      dst );
      cv::imwrite("vis_rgb.png", dst );
    }

    cv::imshow("vis_heightmap", vis_heightmap);
    cv::imwrite("vis_heightmap.png", vis_heightmap);

    cv::imshow("vis_seed", vis_seed);
    cv::imwrite("vis_seed.png", vis_seed);

    cv::imshow("vis_arealimitedflood", vis_arealimitedflood);
    cv::imwrite("vis_arealimitedflood.png", vis_arealimitedflood);

    cv::imshow("vis_rangelimitedflood", vis_rangelimitedflood);
    cv::imwrite("vis_rangelimitedflood.png", vis_rangelimitedflood);
    cv::imshow("vis_onedgesflood", vis_onedgesflood);
    cv::imwrite("vis_onedgesflood.png", vis_onedgesflood);

    cv::imshow("after merge", GetColoredLabel(marker));
    //cv::Mat norm_depth, norm_dist;
    //cv::normalize(depth, norm_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    //cv::imwrite(name_+"depth.png", norm_depth);
    //cv::imwrite(name_+"rgb.png", rgb);
    //cv::imwrite(name_+"heightmap.png", vis_heightmap);
    //cv::imwrite(name_+"seed.png", GetColoredLabel(seedmap));
    //cv::imwrite(name_+"marker.png", GetColoredLabel(marker));
  }
  return true;
}


Segment2DEdgeBased::Segment2DEdgeBased(const std::string& name) 
: Segment2DEdgeBasedAbstract(name) {
}

void Segment2DEdgeBased::SetEdge(const cv::Mat outline_edge,
                                 const cv::Mat convex_edge,
                                 const cv::Mat valid_mask) {
  outline_edge_ = outline_edge;
  convex_edge_ = convex_edge;
  valid_mask_ = valid_mask;
}
void Segment2DEdgeBased::GetEdge(const cv::Mat rgb,
                                 const cv::Mat depth,
                                 cv::Mat& outline_edge,
                                 cv::Mat& convex_edge,
                                 cv::Mat& valid_mask,
                                 bool verbose){
  outline_edge = outline_edge_;
  convex_edge = convex_edge_;
  valid_mask  = valid_mask_;
  return;
}

void Segment2Dthreshold::GetEdge(const cv::Mat rgb,
                                 const cv::Mat depth,
                                 cv::Mat& outline_edge,
                                 cv::Mat& convex_edge,
                                 cv::Mat& valid_mask,
                                 bool verbose){
  // Compute Hessian and NMAS using function of unet_code.cpp.
  assert(false);
  return;
}

Segment2Dthreshold::Segment2Dthreshold(const std::string& name,
                                       double lap_depth_threshold
                                      ) :
Segment2DEdgeBasedAbstract(name),
lap_depth_threshold_(lap_depth_threshold){

}

cv::Mat FilterOutlineEdges(const cv::Mat& outline, bool verbose) {
  cv::Mat labels, stats, centroids;
#if 1
  cv::Mat ones = cv::Mat::ones(5,5,outline.type());
  cv::Mat expanded_outline;
  cv::dilate(outline, expanded_outline, ones);
  cv::threshold(expanded_outline,expanded_outline, 200., 255, cv::THRESH_BINARY);
  cv::connectedComponentsWithStats(expanded_outline,labels,stats,centroids);
#else
  cv::connectedComponentsWithStats(outline,labels,stats,centroids);
#endif

  const int margin = 50;
  std::set<int> inliers;
  for(int i = 0; i < stats.rows; i++){
    // left,top,width,height,area
    const int max_wh = std::max(stats.at<int>(i,2),stats.at<int>(i,3));
    if(max_wh > 50)
      inliers.insert(i);
#if 0
    else{
      // 화면 구석에 위치했는지 여부.
      const int& u0 = stats.at<int>(i,cv::CC_STAT_LEFT);
      const int& v0 = stats.at<int>(i,cv::CC_STAT_TOP);
      const int u1 = u0 + stats.at<int>(i,cv::CC_STAT_WIDTH);
      const int v1 = v0 + stats.at<int>(i,cv::CC_STAT_HEIGHT);
      if(u0 < margin || v0 < margin || u1 > outline.cols-margin || v1 > outline.rows-margin)
        inliers.insert(i);
    }
#endif
  }

  cv::Mat output = cv::Mat::zeros(outline.rows, outline.cols, CV_8UC1);
  for(int r = 0; r < output.rows; r++){
    for(int c = 0; c < output.cols; c++){
      if(outline.at<unsigned char>(r,c) < 200) // dilate에 버그로 작은숫자의 노이즈 있음.
        continue;
      const int& l = labels.at<int>(r,c);
      unsigned char& i = output.at<unsigned char>(r,c);
      i = inliers.count(l)?255:0;
    }
  }

  if(verbose){
    cv::imshow("labels", GetColoredLabel(labels));
    cv::imshow("filtered", output);
  }
  return output;
}


