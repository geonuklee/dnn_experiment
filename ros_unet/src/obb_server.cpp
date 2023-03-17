#include "ros_util.h"
#include "segment2d.h"
#include "mask2obb.h"

#include <vector>
#include <map>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>
#include <cv_bridge/cv_bridge.h>

#include <g2o/types/slam3d/se3quat.h>

//#include "ros_unet/ClearCamera.h"
//#include "ros_unet/SetCamera.h"
#include "ros_unet/ComputeObb.h"
#include <ros/ros.h>
#include <chrono>


class BoxDetector {
public:
  BoxDetector(ros::NodeHandle& nh, const ObbParam& param)
  : nh_(nh), param_(param) {
  }

  void CheckCamera(const std::string& cam_id,
                   const std::string& frame_id,
                   const sensor_msgs::CameraInfo& info){
    if(obb_estimator_.count(cam_id))
      return;
    const std::vector<double>& D = info.D;
    double v_max = 0.;
    for(const double& v : D)
      v_max = std::max<double>(std::abs<double>(v), v_max);

    MarkerCamera camera;

    if(v_max > 1e-5){
      assert(false); // No support yet.
      // TODO newK with cv::getOptimalNewCamera
    } else {
      ROS_INFO_STREAM("No distortion");
      cv::Mat K = cv::Mat::zeros(3,3,CV_32F);
      for(int i = 0; i<K.rows; i++)
        for(int j = 0; j < K.cols; j++)
          K.at<float>(i,j) = info.K.data()[3*i+j];
      cv::Mat D = cv::Mat::zeros(info.D.size(),1,CV_32F);
      for (int j = 0; j < D.rows; j++)
        D.at<float>(j,0) = info.D.at(j);
      cv::Size osize(info.width, info.height);
      camera = MarkerCamera(K,D,osize);
    }

    obb_estimator_[cam_id] = std::make_shared<ObbEstimator>(camera);
    segment2d_[cam_id] = std::make_shared<Segment2DEdgeBased>(cam_id);
    obb_process_visualizer_[cam_id] = std::make_shared<ObbProcessVisualizer>(cam_id, frame_id, nh_);
    pub_xyzrgb_[cam_id] = nh_.advertise<sensor_msgs::PointCloud2>(cam_id+"/xyzrgb",1);
    pub_clouds_[cam_id] = nh_.advertise<sensor_msgs::PointCloud2>(cam_id+"/clouds",1);
    pub_boundary_[cam_id] = nh_.advertise<sensor_msgs::PointCloud2>(cam_id+"/boundary",1);
    pub_vis_mask_[cam_id] = nh_.advertise<sensor_msgs::Image>(cam_id+"/vis_mask",1);
    pub_filteredoutline[cam_id] = nh_.advertise<sensor_msgs::Image>(cam_id+"/vis_filteredoutline",1);
  }

  bool ComputeObb(ros_unet::ComputeObb::Request  &req,
                  ros_unet::ComputeObb::Response &res)
  {
    // Get Tcw from cam_id.
    const std::string& cam_id = req.cam_id.data;
    const std::string& frame_id = req.info.header.frame_id;

    CheckCamera(cam_id, frame_id, req.info);
    auto segment2d = segment2d_.at(cam_id);
    auto obb_estimator = obb_estimator_.at(cam_id);
    auto obb_process_visualizer = obb_process_visualizer_.at(cam_id);

    cv::Mat depth, rgb, convex_edge, outline_edge, valid_mask;
    GetCvMat(req, depth, rgb, convex_edge, outline_edge, valid_mask);
    bool verbose = param_.verbose;

    if(valid_mask.empty()){
      ROS_ERROR_STREAM("Empty valid_mask");
      exit(1);
    }

    const float threshold_depth = .04;
    cv::Mat dd_edge= GetDiscontinuousDepthEdge(depth, threshold_depth);
    cv::bitwise_and(dd_edge, ~convex_edge, dd_edge);
    cv::bitwise_or(outline_edge, dd_edge, outline_edge);

    cv::Mat filtered_outline;
#if 1
    {
      bool verbose_filter = false;
      filtered_outline = FilterOutlineEdges(outline_edge, verbose_filter);
    }
    segment2d->SetEdge(filtered_outline, convex_edge, valid_mask); // TODO
#else
    segment2d->SetEdge(outline_edge, convex_edge, valid_mask);
#endif

    cv::Mat instance_marker;
    std::map<int,int> ins2cls;
    segment2d->Process(rgb, depth, instance_marker, convex_edge, ins2cls, verbose);

    std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr> segmented_clouds, boundary_clouds;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>());


    auto t0 = std::chrono::steady_clock::now();
    obb_estimator->GetSegmentedCloud(rgb,
                                     depth,
                                     instance_marker,
                                     convex_edge,
                                     param_,
                                     segmented_clouds, boundary_clouds, xyzrgb);
    auto t1 = std::chrono::steady_clock::now();
    //std::cout << "elapsed time for segment2d = "
    //  << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << "[ms]" << std::endl;

    obb_estimator->ComputeObbs(segmented_clouds,
                               boundary_clouds,
                               param_,
                               frame_id,
                               obb_process_visualizer
                              );
    res.output = obb_process_visualizer->GetUnsyncedOBB();
    {
      cv_bridge::CvImage cv_img;
      cv_img.encoding = sensor_msgs::image_encodings::TYPE_32SC1;
      cv_img.image    = instance_marker;
      res.marker = *cv_img.toImageMsg();
    }
    {
      cv_bridge::CvImage cv_img;
      cv_img.encoding = sensor_msgs::image_encodings::TYPE_8UC1;
      cv_img.image    = filtered_outline;
      res.filtered_outline = *cv_img.toImageMsg();
    }

    obb_process_visualizer->Visualize();

    // TODO Future works : matching, publixh xyzrgb
    if(pub_filteredoutline.at(cam_id).getNumSubscribers() > 0 && !filtered_outline.empty() ){
      cv_bridge::CvImage msg;
      cv::Mat dst = cv::Mat::zeros(rgb.rows,rgb.cols,CV_8UC3);
      for(int r = 0; r < rgb.rows; r++){
        for(int c = 0; c < rgb.cols; c++){
          auto& pixel = dst.at<cv::Vec3b>(r,c);
          if(filtered_outline.at<unsigned char>(r,c) > 0)
            pixel[2] = 255;
          else if(convex_edge.at<unsigned char>(r,c) > 0)
            pixel[0] = 255;
        }
      }
      cv::addWeighted(rgb, 0.5, dst, 0.5, 0., dst);
      msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      msg.image    = dst;
      pub_filteredoutline.at(cam_id).publish(msg.toImageMsg());
    }
    if(pub_vis_mask_.at(cam_id).getNumSubscribers() > 0) {
      cv::Mat dst = Overlap(rgb, instance_marker,.5);
      cv_bridge::CvImage msg;
      msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      msg.image    = dst;
      pub_vis_mask_.at(cam_id).publish(msg.toImageMsg());
    }
    //if(pub_xyzrgb_.at(cam_id).getNumSubscribers() > 0)
    {
      sensor_msgs::PointCloud2 msg;
      pcl::toROSMsg(*xyzrgb, msg);
      msg.header.frame_id = frame_id;
      pub_xyzrgb_.at(cam_id).publish(msg);
    }

    if(pub_clouds_.at(cam_id).getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      ColorizeSegmentation(segmented_clouds, msg);
      msg.header.frame_id = frame_id;
      pub_clouds_.at(cam_id).publish(msg);
    }
    if(pub_boundary_.at(cam_id).getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      ColorizeSegmentation(boundary_clouds, msg);
      msg.header.frame_id = frame_id;
      pub_boundary_.at(cam_id).publish(msg);
    }

    return true;
  }

private:

  g2o::SE3Quat GetSE3Quat(const geometry_msgs::Pose& pose) const {
    const auto& p = pose.position;
    const auto& q = pose.orientation;
    Eigen::Quaterniond quat(q.w,q.x,q.y,q.z);
    Eigen::Vector3d t(p.x,p.y,p.z);
    g2o::SE3Quat se3quat(quat.toRotationMatrix(), t);
    return se3quat;
  }

  void GetCvMat(ros_unet::ComputeObb::Request& req,
                cv::Mat& depth,
                cv::Mat& rgb,
                cv::Mat& convex_edge,
                cv::Mat& outline_edge,
                cv::Mat& valid_mask
                ) {
    cv::Mat odepth = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    cv::Mat orgb = cv_bridge::toCvCopy(req.rgb, sensor_msgs::image_encodings::TYPE_8UC3)->image;
    cv::Mat och[2]; {
      cv::Mat edges = cv_bridge::toCvCopy(req.edge, sensor_msgs::image_encodings::TYPE_8UC2)->image;
      cv::split(edges, och);
    }

    valid_mask = cv_bridge::toCvCopy(req.p_mask, sensor_msgs::image_encodings::TYPE_32SC1)->image;
    valid_mask = valid_mask < 1;

    if(mx_.empty()){
      depth = odepth;
      rgb = orgb;
      convex_edge = och[1];
      outline_edge = och[0] == 1;
    }
    else{
      assert(false); // No support yet.
    }
    return;
  }

  ros::NodeHandle& nh_;

  ObbParam param_;
  std::map<std::string, std::shared_ptr<Segment2DEdgeBased> > segment2d_;
  std::map<std::string, std::shared_ptr<ObbEstimator> > obb_estimator_;
  std::map<std::string, std::shared_ptr<ObbProcessVisualizer> > obb_process_visualizer_;
  std::map<std::string, ros::Publisher> pub_xyzrgb_, pub_clouds_, pub_boundary_, pub_vis_mask_,
    pub_filteredoutline;
  cv::Mat mx_, my_;
};


int main(int argc, char **argv) {
  ros::init(argc, argv, "~");
  ros::NodeHandle nh("~");
  ObbParam param;
  //std::vector<int> cameras = GetRequiredParamVector<int>(nh, "cameras");
  param.min_points_of_cluster = GetRequiredParam<double>(nh, "min_points_of_cluster");
  param.voxel_leaf = GetRequiredParam<double>(nh, "voxel_leaf");
  param.euclidean_filter_tolerance = GetRequiredParam<double>(nh, "euc_tolerance"); // [meter] small tolerance condisering bg invasion
  param.verbose = GetRequiredParam<bool>(nh, "verbose");
  BoxDetector box_detector(nh, param);

  ros::ServiceServer s2
    = nh.advertiseService("ComputeObb", &BoxDetector::ComputeObb, &box_detector);

  ros::spin();
  return 0;
}
