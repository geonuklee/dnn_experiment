uint8 NONE            = 0
uint8 MOVE            = 1
uint8 SUCTION         = 2
uint8 CLAMP           = 3
uint8 CONVEYOR        = 4
uint8 SUCTION_DOWN    = 5
uint8 CLAMP_DOWN      = 6
uint8 CONVEYOR_DOWN   = 7
uint8 MOVE_FORWARD    = 200

uint8 NO_INFO         = 0
uint8 ALIGNED         = 1
uint8 NOT_ALIGNED     = 2

std_msgs/Header header

uint8 arm0_type
uint8 arm1_type

bool arm0_caution
bool arm1_caution

geometry_msgs/Pose arm0_pose
geometry_msgs/Pose arm1_pose

float32 conv_height

bool cam0_tilt
bool cam1_tilt
bool move_forward

uint8 stack_option
int32[5] obj_kind
int32[5] arm0_target
int32[5] arm1_target