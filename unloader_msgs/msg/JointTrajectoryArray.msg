uint8 UNKNOWN  = 0
uint8 MOVE     = 1
uint8 SUCTION  = 2
uint8 CLAMP    = 3
uint8 CONVEYOR = 4

std_msgs/Header header

uint8 motion_type
trajectory_msgs/JointTrajectory[] trajectories
unloader_msgs/BoolArray active_suction
unloader_msgs/BoolArray active_clamp
unloader_msgs/BoolArray active_conv