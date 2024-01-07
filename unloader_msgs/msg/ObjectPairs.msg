# This message represents correspondances between objects as a pair
# of lists: reference and destination. All objects are represented
# by an integer ID contained in each reference and destination.
# IDs at the same index in both lists (reference, destination) 
# are a pair, this means, they represented the same object. 
# Thus, reference and destination lists should have the same length
std_msgs/Header header
uint32[] metadata 
int32[] reference
int32[] destination