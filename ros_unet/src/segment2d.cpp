#include "segment2d.h"
#include "ros_util.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgproc.hpp>
#include <numeric>

MarkerCamera::MarkerCamera(const cv::Mat& K,
                           const cv::Mat& D,
                           const cv::Size& image_size)
: K_(K), D_(D), image_size_(image_size)
{
}


sensor_msgs::CameraInfo MarkerCamera::AsCameraInfo() const {
  sensor_msgs::CameraInfo info;
  //info.K.reserve(9);
  int i = 0;
  for(int r = 0; r < 3; r++)
    for(int c = 0; c < 3; c++)
      info.K[i++] = K_.at<float>(r,c);
  info.D.reserve(D_.rows);
  for(int r = 0; r < D_.rows; r++)
    info.D.push_back(D_.at<float>(r,0));
  info.height = image_size_.height;
  info.width = image_size_.width;
  return info;
}

void Segment2DAbstract::Rectify(cv::Mat given_rgb,
                                cv::Mat given_depth
                               ) {
  cv::remap(given_rgb, rectified_rgb_, map1_, map2_, cv::INTER_NEAREST);
  cv::remap(given_depth, rectified_depth_, map1_, map2_, cv::INTER_NEAREST);
}

int Convert(const std::map<int,int>& convert_lists,
             const std::set<int>& leaf_seeds,
             const cv::Mat& filtered_edge, 
             const cv::Mat& depthmask,
             cv::Mat& seed ){
  int bg_idx = std::max<int>(100, (*leaf_seeds.rbegin()) + 1);
  int* ptr = (int*) seed.data;
  const int n = seed.rows*seed.cols;

  // Profile 결과, 연산의 병목지점은 laplacian of depth(약40%), seed_contour(약 50%) 이므로 그 이후작업의 성능개선은 무의미.
  for(int r = 0; r < n; r++){
      int& idx = ptr[r];
      if(filtered_edge.data[r] > 0)
        idx = 1;
      else if(depthmask.data[r] < 1)
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

Segment2DAbstract::Segment2DAbstract(const sensor_msgs::CameraInfo& camera_info, cv::Size compute_size, std::string name)
: name_(name)
{

  cv::Mat K = cv::Mat::zeros(3,3,CV_32F);
  for(int i = 0; i<K.rows; i++)
    for(int j = 0; j < K.cols; j++)
      K.at<float>(i,j) = camera_info.K.data()[3*i+j];
  cv::Mat D = cv::Mat::zeros(camera_info.D.size(),1,CV_32F);
  for (int j = 0; j < D.rows; j++)
    D.at<float>(j,0) = camera_info.D.at(j);

  cv::Size osize(camera_info.width, camera_info.height);
  cv::Mat newK = cv::getOptimalNewCameraMatrix(K, D, osize, 1., compute_size);
  camera_ = MarkerCamera(newK,cv::Mat::zeros(4,1,CV_32F), compute_size);

  cv::initUndistortRectifyMap(K, D, cv::Mat::eye(3,3,CV_32F),
                              camera_.K_, camera_.image_size_, CV_32F, map1_, map2_);
}

Segment2DEdgeBased::Segment2DEdgeBased(const sensor_msgs::CameraInfo& camera_info, cv::Size compute_size, std::string name)
  : Segment2DAbstract(camera_info,compute_size, name)
{
  cv::Mat src = 255*cv::Mat::ones(compute_size, CV_8UC1);
  cv::rectangle(src, cv::Point(0,0), cv::Point(compute_size.width,compute_size.height), 0, 50);
  src.convertTo(vignett32S_, CV_32SC1);
  src.convertTo(vignett8U_, CV_8UC1);
}

cv::Mat GetDepthMask(const cv::Mat depth) {
  cv::Mat depthmask;
#if 1
  // TODO Remove hard code considering std_2021.08. experiment.
  depthmask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
  for(int r = 0; r < depth.rows; r++){
    for(int c = 0; c < depth.cols; c++){
      const float& d = depth.at<float>(r, c);
      if(d > 0.01 && d < 3.)
          depthmask.at<unsigned char>(r,c) = 255;
    }
  }
#else
  cv::threshold(depth, depthmask, 0.1, 255, cv::THRESH_BINARY);
#endif

  const cv::Mat erode_kernel = cv::Mat::ones(5, 5, CV_32F);
  cv::erode(depthmask, depthmask, erode_kernel, cv::Point(-1,-1), 1);
  cv::threshold(depthmask, depthmask, 200, 1, cv::THRESH_BINARY);

  depthmask.convertTo(depthmask, CV_8UC1);
  return depthmask;
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

bool Segment2DEdgeBased::Process(const cv::Mat rgb,
                                 const cv::Mat given_depth,
                                 cv::Mat& marker,
                                 cv::Mat& convex_edge,
                                 cv::Mat& depthmap,
                                 std::map<int,int>& instance2class,
                                 bool verbose){
  Rectify(rgb, given_depth);
  depthmap = GetRectifiedDepth();
  cv::Mat rectified_rgb = GetRectifiedRgb();

  if(vignett32S_.empty() ){
    vignett32S_ = 255*cv::Mat::ones(rectified_rgb.size(), CV_32SC1);
    vignett8U_  = 255*cv::Mat::ones(rectified_rgb.size(), CV_8UC1);
  }

  return _Process(rectified_rgb, depthmap, marker, convex_edge, instance2class, verbose);
}

bool Segment2DEdgeBased::_Process(cv::Mat rgb,
                        cv::Mat depth,
                        cv::Mat& marker,
                        cv::Mat& convex_edge,
                        std::map<int,int>& instance2class,
                        bool verbose){
  cv::Mat depthmask = GetDepthMask(depth);
  cv::Mat validmask = GetValidMask(depthmask);
  cv::Mat outline_edge, surebox_mask;
  GetEdge(rgb, depth, validmask, outline_edge, convex_edge, surebox_mask, verbose);
  if(outline_edge.empty())
    return false;

  if(!surebox_mask.empty()) { // pre-filtering for surebox
    cv::Mat sureground;
    cv::bitwise_or(outline_edge > 0, surebox_mask > 0, sureground);
    cv::Mat element5(3, 3, CV_8U, cv::Scalar(1));
    cv::morphologyEx(sureground, sureground, cv::MORPH_CLOSE, element5);

    cv::bitwise_and(validmask, sureground,  validmask);
    //cv::imshow("sureground", sureground*255);
  }

  if(! vignett32S_.empty() )
    cv::bitwise_and(outline_edge, vignett8U_, outline_edge);


  cv::Mat divided;
  cv::Mat dist_transform; {
    cv::bitwise_and(depthmask, ~outline_edge, divided);
    cv::bitwise_and(vignett8U_, divided, divided);

    if(divided.type()!=CV_8UC1)
      divided.convertTo(divided, CV_8UC1); // distanceTransform asks CV_8UC1 input.

    cv::distanceTransform(divided, dist_transform, cv::DIST_L2, cv::DIST_MASK_3);
  }

  cv::Mat seed = cv::Mat::zeros(camera_.image_size_, CV_32SC1);
  cv::Mat seed_contours;
  std::set<int> leaf_seeds;
  int bg_idx = -1;
  {
    const int mode   = cv::RETR_TREE; // RETR_CCOMP -> RETR_EXTERNAL
    const int method = cv::CHAIN_APPROX_SIMPLE;

    int n = 15;
    double dth = camera_.image_size_.width * 0.006;
    float min_width = 20.;

    std::map<int,std::set<int> > seed_childs;
    std::map<int,int> seed_parents;

    int idx = 1;
    for(int i = 1; i < n; i++){
      int th_distance = dth*i;
      cv::Mat local_seed;
      cv::threshold(dist_transform, local_seed, th_distance, 255, cv::THRESH_BINARY);
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
        const int& exist_idx = seed.at<int>(y,x);
        auto ar = cv::minAreaRect(cnt);
        if(std::min(ar.size.width,ar.size.height) < min_width)
          continue;
        idx += 1;
        n_insertion++;
        seed_childs[exist_idx].insert(idx);
        seed_parents[idx] = exist_idx;
        std::vector<std::vector<cv::Point> > cnts = { cnt, };
        cv::drawContours(seed, cnts, 0, idx,-1);
        seed_childs[idx];
      }
      if(n_insertion < 1)
        break;
    }

    seed_contours = seed.clone();
    for(const auto& it : seed_childs){
      const int& parent = it.first;
      const std::set<int>& childs = it.second;
      if(parent == 0){
        for(const int& child : childs)
          if(seed_childs.at(child).empty())
            leaf_seeds.insert(child);
      }
      else if(childs.size() == 1){
        int child = *childs.begin();
        if(seed_childs.at(child).empty())
          leaf_seeds.insert(child);
      }
      else if(childs.empty() )
        leaf_seeds.insert(parent);
    }

    std::map<int, int> convert_lists;
    std::set<int> copied_leafs = leaf_seeds;

    idx = -1;
    for(const int& idx : copied_leafs){
      int keyidx = idx;
      int parent = -1;
      std::set<int> non_sibiling;
      while(true){
        parent = seed_parents.at(keyidx);
        const auto& siblings = seed_childs.at(parent);
        if(parent == 0){
          non_sibiling.insert(keyidx);
          break;
        }
        else if(siblings.size() == 1)
          non_sibiling.insert(keyidx);
        else{
          non_sibiling.insert(keyidx);
          break;
        }
        keyidx = parent;
      }
      int root = *non_sibiling.begin(); // root = min(non_sibilig)
      if(idx == root)
        continue;
      leaf_seeds.insert(root);
      leaf_seeds.erase(idx);
      for(const int& i : non_sibiling){
        if(i != root)
          convert_lists[i] = root;
      }
    } // compute leaf_seeds

    if(!leaf_seeds.empty()) {
      // Convert elements of seed from exist_idx to convert_lists
      bg_idx = Convert(convert_lists, leaf_seeds,
                       outline_edge, depthmask,
                       seed);
    }
    else
      bg_idx = 2;
  }

  cv::Mat shape_marker = seed.clone();
  cv::watershed(rgb, shape_marker);

  {
    cv::Mat marker2 = shape_marker.clone();
    for(int r = 0; r < marker2.rows; r++){
      for(int c = 0; c < marker2.cols; c++){
        int& v = marker2.at<int>(r,c);
        if(v <= 1)
          v = 0; // to unknown.
      }
    }
    cv::watershed(rgb, marker2);
    for(int r = 0; r < marker2.rows; r++){
      for(int c = 0; c < marker2.cols; c++){
        int& v = marker2.at<int>(r,c);
        if(v == bg_idx)
          v = 0;
        else if(v <=1)
          v = 0;
      }
    }
    marker = marker2;
  }

  if(verbose){
    cv::Mat dst = Overlap(rgb, marker);
    //cv::imshow(name_+"seed contour", GetColoredLabel(seed_contours));
    //cv::imshow(name_+"seed", GetColoredLabel(seed) );
    //cv::imshow(name_+"shape_marker", GetColoredLabel(shape_marker) );
    //cv::imshow(name_+"final_marker", GetColoredLabel(marker) );
    // cv::imshow(name_+"groove", 255*groove );
    //cv::flip(dst,dst,0);
    //cv::flip(dst,dst,1);
    //cv::imshow(name_+"dst", dst);
    cv::imshow(name_+"outline_edge.png", 255*outline_edge);

    cv::Mat norm_depth, norm_dist;
    cv::normalize(depth, norm_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::normalize(dist_transform, norm_dist, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(name_+"dst.png", dst );
    cv::imwrite(name_+"depth.png", norm_depth);
    cv::imwrite(name_+"rgb.png", rgb);
    cv::imwrite(name_+"outline_edge.png", 255*outline_edge);
    cv::imwrite(name_+"valid.png", 255*validmask);
    cv::imwrite(name_+"dist.png", dist_transform);
    cv::imwrite(name_+"norm_dist.png", norm_dist);
    cv::imwrite(name_+"seed_contour.png", GetColoredLabel(seed_contours));
    cv::imwrite(name_+"seed.png", GetColoredLabel(seed));
    cv::imwrite(name_+"marker.png", GetColoredLabel(marker));
  }
  return true;
}

void Segment2Dthreshold::GetEdge(const cv::Mat rgb,
                                 const cv::Mat depth,
                                 const cv::Mat validmask,
                                 cv::Mat& outline_edge,
                                 cv::Mat& convex_edge,
                                 cv::Mat& surebox_mask,
                                 bool verbose){
  // TODO Compute Hessian and NMAS using function of unet_code.cpp.
  assert(false);
  return;
}

Segment2Dthreshold::Segment2Dthreshold(const sensor_msgs::CameraInfo& camerainfo,
                                       cv::Size compute_size,
                                       std::string name,
                                       double lap_depth_threshold
                                      ) :
Segment2DEdgeBased(camerainfo, compute_size, name),
lap_depth_threshold_(lap_depth_threshold){

}

