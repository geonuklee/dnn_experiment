uint8 SUCTION  = 0
uint8 CLAMP    = 1
uint8 CONVEYOR = 2

std_msgs/Header header
uint64 tfinish
unloader_msgs/Object[] objects
uint64[] erased_objects
uint8 motion_type
