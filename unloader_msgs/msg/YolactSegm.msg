
# std_msgs/Header header
#  - seq: image sequence (image number)
#  - frame_id: camera number
# 	
# uint64[] image_size
#  - list of integers containing: [width, heigth, depth]
#
# uint64[] num_objs
#  - total number of objects detected
#
# uint64[] segm_masks
#  - list containing all the masks. Each mask must be represented as 
#    a polygon, this is, as an nx2 array where there are n points each
#    each described by a pair of x,y coordinates. This can be obtained
#    using openCV's findContour. All of these contours should be then 
#    stored in a 1-d list. This implies that the start and end of each
#    contour will be unknown. To recover the length of each contour
#    the number of points n of each polygon
#    should be stored using the 'mask_len' field.
#
# uint64[] mask_len
#  - list containing the number of points of each polygon. 
#
# uint64[] class_id
#  - list of all class_ids corresponding to each mask in segm_masks. 
#    There should be as many class_ids as num_objs
#
# uint64[] score
#  - list of all scores corresponding to each mask in segm_masks. 
#    There should be as many scores as num_objs
#
# uint64[] bboxes
#  - list of the x_min,y_min,x_max,y_max coordinates for the bounding
#    box for each of the masks in segm_masks. Because each box has 
#    4 corners, there should be 4*num_objs coordinates.

std_msgs/Header header 
uint64[] image_size
uint64 num_objs
uint64[] segm_masks
uint64[] mask_len
uint64[] class_id
float32[] score
uint64[] bboxes