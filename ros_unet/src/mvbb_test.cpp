#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

namespace PMP = CGAL::Polygon_mesh_processing;
typedef CGAL::Exact_predicates_inexact_constructions_kernel    K;
typedef K::Point_3                                             Point;
typedef CGAL::Surface_mesh<Point>                              Surface_mesh;

#include <ros/ros.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "~");
  ros::NodeHandle nh("~");
  // GetRequiredParam<double>(nh, "min_points_of_cluster");
  // GetRequiredParam<double>(nh, "voxel_leaf");

  int hz = 1;
  ros::Rate rate(hz);

  while(!ros::isShuttingDown()){
    printf("Hello CGAL world\n");


    rate.sleep();
    ros::spinOnce();
  }
  return 0;
}
