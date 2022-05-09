#ifndef MASK2OBB
#define MASK2OBB

#include <memory>
#include <opencv2/highgui.hpp>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <g2o/types/slam3d/se3quat.h>
#include <string>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>

#include <geometry_msgs/PoseArray.h>
#include <unloader_msgs/Object.h>
#include <unloader_msgs/ObjectArray.h>

#include "segment2d.h"

/**
@struct ObbParam
@brief The structure is a set of parameters required for ComputeBoxOBB().
*/
struct ObbParam{

  /**
  @brief The alpha parameter of concavehull computation.
  The parameter is a positive, non-zero value, defining the maximum length from a vertex to the facet center (center of the voronoi cell).
  <a href="https://pointclouds.org/documentation/classpcl_1_1_concave_hull.html#ab2e24dc457ff32634aa2549a10bc7562">See this</a>.
  */
  float concavehull_alpha_ = 0.1;

  /**
  @brief The maximum distance error distance[meter] between plane and inliner points.
  It is required by ComputeBoxOBB().
  */
  float max_surface_error_ = 0.02;
  /**
  @brief The instance id restarts from 0 since 'maximum_instance_id'.
  */
  int maximum_instance_id = 10000;

  /**
  @brief The tolerance[meter] for the euclidean filter.
  */
  double euclidean_filter_tolerance;

  /**
  @brief The minumum number of points for each cluster.
  It is referenced by EuclideanFilterXYZ() and ComputeBoxOBB().
  */
  size_t min_points_of_cluster;


  /**
  @brief The minimum inner product between noraml direction of front plane and depth direction.
  */
  double min_cos_dir = std::cos( 80. /180. * M_PI);

  /**
  */
  double search_ratio = 0.2;

  /**
  @brief The minimum match tolerance[meter] for matching 'OBB'.
  */
  double match_tolerance = 0.1;
  double min_iou = 0.2;

  double max_view_tan = 2.;
  enum MATCHMETHOD{
    WITHOUT_IOU=0,
    WITH_IOU=1,
  };

  enum MATCHMODE{
    NO_MATCH=0,
    CURRENT_ONLY=1,
    TRACKING=2
  };

  enum SENSORMODEL{
    HELIOS=0,
    K4A=1
  };

  MATCHMODE match_mode;
  MATCHMETHOD match_method;
  SENSORMODEL sensor_model;

  double min_z_floor; // Get it from rosparam
  double min_visible_ratio_of_frontplane = 0.;
  double voxel_leaf;  // voxel leaf size of input points.
  double forcing_iou_method_z_threshold = -0.4;
  bool verbose;
};

void ColorizeSegmentation(const std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& clouds,
                          sensor_msgs::PointCloud2& msg);

class ObbProcessVisualizer{
public:
  /**
  @brief The initialization without input parameter.
  */
  ObbProcessVisualizer() :cam_id_(""){ };

  /**
  @brief The initialization.
  @param[in] cam_id The index of camera.
  @param[in, out] nh The ros node handle is required to create ros topic publisher.
  */
  ObbProcessVisualizer(const std::string& cam_id, ros::NodeHandle& nh);

  /**
  @brief It publish rostopic for segmented boundary points with distinctive colormap for unsynced instance id.
  */
  void VisualizeBoundaryClouds(const std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds);
  /**
  @brief It publish rostopic for segmented (inner) cloud points with distinctive colormap for unsynced instance id.
  */
  void VisualizeClouds(const std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds);

  /**
  @brief It collects contour marker which will be published by ObbProcessVisualizer::Visualize().
  */
  void PutContour(const visualization_msgs::Marker& contour_marker);
  /**
  @brief It collects pose0 marker which will be published by Visualize().
  Denote that Pose0 is not a final pose of OBB.
  */
  void PutPose0(const geometry_msgs::Pose& pose0);
  /**
  @brief It collects pose marker which will be published by ObbProcessVisualizer::Visualize().
  The topic is helpful check the orientation of OBB because it shows xyz axis as an rgb axis.
  */
  void PutPose(const geometry_msgs::Pose& pose);
  /**
  @brief It collects unsynced OBB marker which will be published by ObbProcessVisualizer::Visualize().
  */
  void PutUnsyncedOBB(const visualization_msgs::Marker& unsynced_obb);
  
  visualization_msgs::MarkerArray GetUnsyncedOBB();

  /**
  @brief It publish collected topics at once to show intermediate process.
  */
  void Visualize();

private:
  const std::string cam_id_;
  ros::Publisher pub_mask;
  ros::Publisher pub_boundary;
  ros::Publisher pub_clouds;
  ros::Publisher pub_contour;
  ros::Publisher pub_pose0;
  ros::Publisher pub_pose;
  ros::Publisher pub_unsynced_obb;

  sensor_msgs::PointCloud2 Convert(const std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds);

  visualization_msgs::MarkerArray contours_old_, contours_current_;
  geometry_msgs::PoseArray pose0_array_;
  geometry_msgs::PoseArray pose_array_;
  visualization_msgs::MarkerArray obb_current_, obb_old_;
};


class ObbEstimator {
public:
  /**
  @param[in] camera_info The camera information..
  */
  ObbEstimator() {};
  ObbEstimator(const MarkerCamera& marker_camera);

  /**
  @brief It computes segmented points cloud based 2D instance segmentation mask.

  The points cloud projected on non-boundary pixels are classified '(inner points) cloud',
  while the points cloud which are unprojected from boundary pixels are defined as 'boundary (points)'.

  @param[in] Tcw The SE(3) transformation to rgb {c}amera from {w}orld.
  @param[in] given_points The points cloud from sensor on {w}orld coordinate.
  @param[in] rgb
  @param[in] depth
  @param[out] segmented_clouds The maps from yolact instance id to segmented (inner points) cloud.
  @param[out] boundary_clouds  The maps from yolact instance id to segmented boundary (points) cloud.
  */
  void GetSegmentedCloud( const g2o::SE3Quat& Tcw,
                         cv::Mat rgb,
                         cv::Mat depth,
                         cv::Mat instance_marker,
                         cv::Mat convex_edge,
                         const ObbParam& param,
                         std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& segmented_clouds,
                         std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& boundary_clouds,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzrgb
                        );

  void ComputeObbs(const std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& clouds,
                   const std::map<int, pcl::PointCloud<pcl::PointXYZLNormal>::Ptr>& boundary_clouds,
                   const ObbParam& param,
                   const g2o::SE3Quat& Tcw,
                   const std::string& cam_id,
                   std::shared_ptr<ObbProcessVisualizer> visualizer
                  );

protected:
  cv::Point2f GetUV(int r, int c) const;
  cv::Mat nu_, nv_;
};



#endif
