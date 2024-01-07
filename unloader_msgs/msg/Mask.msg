int32 width # total mask width
int32 height # total mask height
int32[] crop_roi # x1,y1,w,h format crop roi of mask. only positive area is cropped and encoded to the bitset. 
uint8[] mask # Mask encoded as bitset
