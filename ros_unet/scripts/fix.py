from os import path as osp
import glob2
import pickle

def fix():
    datasets = glob2.glob('/home/geo/catkin_ws/src/ros_unet/obb_dataset_*')
    for dataset_dir in datasets:
        usage = osp.basename(dataset_dir).split('_')[-1]
        files = glob2.glob(osp.join(dataset_dir, '*.pick'))
        for fn in files:
            with open(fn, 'rb') as f:
                pick = pickle.load(f)
                if 'fullfn' in pick:
                    fullfn = pick.pop('fullfn')
                    pick['rosbag_fn'] = osp.join('rosbag_%s'%usage, osp.basename(fullfn) )
                cvgt_fn0 = pick['cvgt_fn']
                if 'home' in 'home' in cvgt_fn0.split('/')[:2]:
                    cvgt_fn = osp.basename(cvgt_fn0)
                    pick['cvgt_fn'] = osp.join('obb_dataset_%s'%usage, cvgt_fn)

            with open(fn, 'wb') as f:
                pickle.dump(pick, f, protocol=2)

from unet.segment_dataset import *

def test():
    dataset = ObbDataset('obb_dataset_alignedroll',max_frame_per_scene=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for j, data in enumerate(dataloader):
        print(j, data['rgb'].shape)
    pass

if __name__ == '__main__':
    #fix()
    test()
