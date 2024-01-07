uint8 UNKNOWN = 0
uint8 BOX     = 1
uint8 ICEBOX  = 2
uint8 POUCH   = 3
uint8 SACK    = 4
uint8 BOTTLE  = 5

uint8 NONE    = 0
uint8 CJ      = 1
uint8 FRAGILE = 2
uint8 UPDOWN  = 4

uint8 type
uint32 instance_id
uint8 caution

bool[6] visible_plane
int32[] yolact_instance_id

geometry_msgs/Pose center_pose
geometry_msgs/Vector3 size

sensor_msgs/PointCloud2 point_cloud
