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

#include "ros_unet/ClearCamera.h"
#include "ros_unet/SetCamera.h"
#include "ros_unet/ComputeObb.h"
#include <ros/ros.h>

class BoxDetector {
public:
  BoxDetector(ros::NodeHandle& nh, const ObbParam& param)
  : nh_(nh), param_(param) {
  }

  bool ClearCamera(ros_unet::ClearCamera::Request  &req,
                 ros_unet::ClearCamera::Response &res){

    return true;
  }

  bool SetCamera(ros_unet::SetCamera::Request  &req,
                 ros_unet::SetCamera::Response &res)
  {
    std::cout << "Set camera" << std::endl;
    const std::vector<double>& D = req.info.D;
    double v_max = 0.;
    for(const double& v : D)
      v_max = std::max<double>(std::abs<double>(v), v_max);

    MarkerCamera camera;

    if(v_max > 1e-5){
      assert(false); // No support yet.
      // TODO newK with cv::getOptimalNewCamera
    } else {
      std::cout << "No distortion " << std::endl;
      cv::Mat K = cv::Mat::zeros(3,3,CV_32F);
      for(int i = 0; i<K.rows; i++)
        for(int j = 0; j < K.cols; j++)
          K.at<float>(i,j) = req.info.K.data()[3*i+j];
      cv::Mat D = cv::Mat::zeros(req.info.D.size(),1,CV_32F);
      for (int j = 0; j < D.rows; j++)
        D.at<float>(j,0) = req.info.D.at(j);
      cv::Size osize(req.info.width, req.info.height);
      camera = MarkerCamera(K,D,osize);
    }

    const std::string& cam_id = req.cam_id.data;
    obb_estimator_[cam_id] = std::make_shared<ObbEstimator>(camera);
    segment2d_[cam_id] = std::make_shared<Segment2DEdgeBased>(cam_id);
    obb_process_visualizer_[cam_id] = std::make_shared<ObbProcessVisualizer>(cam_id, nh_);
    pub_clouds_[cam_id] = nh_.advertise<sensor_msgs::PointCloud2>(cam_id+"/clouds",1);
    pub_boundary_[cam_id] = nh_.advertise<sensor_msgs::PointCloud2>(cam_id+"/boundary",1);
    pub_vis_mask_[cam_id] = nh_.advertise<sensor_msgs::Image>(cam_id+"/vis_mask",1);
    return true;
  }

  bool ComputeObb(ros_unet::ComputeObb::Request  &req,
                  ros_unet::ComputeObb::Response &res)
  {
    // Get Tcw from cam_id.
    const std::string& cam_id = req.cam_id.data;
    auto segment2d = segment2d_.at(cam_id);
    auto obb_estimator = obb_estimator_.at(cam_id);
    auto obb_process_visualizer = obb_process_visualizer_.at(cam_id);

    g2o::SE3Quat Tcw = GetSE3Quat(req.Twc).inverse();

    cv::Mat depth, rgb, convex_edge, outline_edge, surebox;
    GetCvMat(req, depth, rgb, convex_edge, outline_edge, surebox);
    segment2d->SetEdge(outline_edge, convex_edge, surebox);

    cv::Mat instance_marker;
    std::map<int,int> ins2cls;
    bool verbose = false;
    segment2d->Process(rgb, depth, instance_marker, convex_edge, ins2cls, verbose);

    //std::cout << "Compute OBB" << std::endl;

    std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr> segmented_clouds, boundary_clouds;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb;

    obb_estimator->GetSegmentedCloud(Tcw,
                                     rgb,
                                     depth,
                                     instance_marker,
                                     convex_edge,
                                     param_,
                                     segmented_clouds, boundary_clouds, xyzrgb);

    obb_estimator->ComputeObbs(segmented_clouds,
                               boundary_clouds,
                               param_,
                               Tcw,
                               cam_id,
                               obb_process_visualizer
                              );
    res.output = obb_process_visualizer->GetUnsyncedOBB();
    obb_process_visualizer->Visualize();

    // TODO Future works : matching, publixh xyzrgb
    if(pub_vis_mask_.at(cam_id).getNumSubscribers() > 0) {
      cv::Mat dst = Overlap(rgb, instance_marker);
      cv_bridge::CvImage msg;
      msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
      msg.image    = dst;
      pub_vis_mask_.at(cam_id).publish(msg.toImageMsg());
    }
    if(pub_clouds_.at(cam_id).getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      ColorizeSegmentation(segmented_clouds, msg);
      pub_clouds_.at(cam_id).publish(msg);
    }
    if(pub_boundary_.at(cam_id).getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 msg;
      ColorizeSegmentation(boundary_clouds, msg);
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
                cv::Mat& surebox
                ) {
    cv::Mat odepth = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
    cv::Mat orgb = cv_bridge::toCvCopy(req.rgb, sensor_msgs::image_encodings::TYPE_8UC3)->image;
    cv::Mat och[2]; {
      cv::Mat edges = cv_bridge::toCvCopy(req.edge, sensor_msgs::image_encodings::TYPE_8UC2)->image;
      cv::split(edges, och);
    }

    if(mx_.empty()){
      depth = odepth;
      rgb = orgb;
      convex_edge = och[1];
      outline_edge = och[0] == 1;
      surebox = och[0] == 2;
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
  std::map<std::string, ros::Publisher> pub_clouds_, pub_boundary_, pub_vis_mask_;
  ros::Publisher pub_xyzrgb; // TODO <<- 이건 각 카메라 별이 아니라, 전체 카메라 묶어서.
  cv::Mat mx_, my_;
};


int main(int argc, char **argv) {
  ros::init(argc, argv, "~");
  ros::NodeHandle nh("~");
  ObbParam param;
  //std::vector<int> cameras = GetRequiredParamVector<int>(nh, "cameras");
  param.min_z_floor = GetRequiredParam<double>(nh, "min_z_floor");
  param.min_points_of_cluster = GetRequiredParam<double>(nh, "min_points_of_cluster");
  param.voxel_leaf = GetRequiredParam<double>(nh, "voxel_leaf");
  param.euclidean_filter_tolerance = 2.* param.voxel_leaf;
  param.verbose = GetRequiredParam<bool>(nh, "verbose");
  BoxDetector box_detector(nh, param);

  ros::ServiceServer s0
    = nh.advertiseService("SetCamera", &BoxDetector::SetCamera, &box_detector);

  ros::ServiceServer s1
    = nh.advertiseService("ClearCamera", &BoxDetector::ClearCamera, &box_detector);


  ros::ServiceServer s2 
    = nh.advertiseService("ComputeObb", &BoxDetector::ComputeObb, &box_detector);

  ros::spin();
  return 0;
}
