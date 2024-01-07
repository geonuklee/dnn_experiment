uint8 NONE            = 0
uint8 INSPECTION_CONT = 1
uint8 INSPECTION_END  = 2
uint8 CALIB_CONT      = 3
uint8 CALIB_END       = 4

std_msgs/Header header

bool alive
bool ready
bool heavy_alarm
bool light_alarm
bool busy

uint8 sequence
bool motion_request
unloader_msgs/MultiarmPose dual_arms
unloader_msgs/JointTrajectoryArray joint_trajectory