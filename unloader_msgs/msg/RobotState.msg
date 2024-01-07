std_msgs/Header header

bool alive
bool ready
bool heavy_alarm
bool light_alarm
bool busy
bool moving

bool motion_confirm

sensor_msgs/JointState joint_state
unloader_msgs/BoolArray suction_active
unloader_msgs/BoolArray clamp_active
unloader_msgs/BoolArray conv_active