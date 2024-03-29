from os import path as osp
import pickle
import numpy as np
import copy
from os import makedirs
import glob2
from sys import version_info
from unet_ext import UnprojectPointscloud

class Data_Configs:
    #sem_names = ['bg', 'box']
    #sem_ids = [-1, 0]
    sem_names = ['box']
    sem_ids = [0]


    points_cc = 9
    sem_num = len(sem_names)
    ins_max_num = 50 # 24 for ...
    train_pts_num =8000 
    test_pts_num = 8000

def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        if version_info.major == 3:
            return np.concatenate([data, dup_data], 0), [n for n in range(N)]+list(sample)
        else:
            return np.concatenate([data, dup_data], 0), range(N)+list(sample)

def sample_data_label(data, label, inslabel, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    new_inslabel = inslabel[sample_indices]
    return new_data, new_label, new_inslabel


def xyz_normalize(xyz, mins, wvec):
    return (xyz - mins)/wvec

def get_blocks(xyzrgb, ins_points, sem_points, num_points=1000):
    n = 1
    if n == 1:
        pc_sampled, sem_sampled, ins_sampled\
                = sample_data_label(xyzrgb, sem_points, ins_points, num_points)
        return np.expand_dims(pc_sampled,0), np.expand_dims(sem_sampled,0), np.expand_dims(ins_sampled,0)
    lim0 = np.amin(xyzrgb[:,:2], axis=0)
    lim1 = np.amax(xyzrgb[:,:2], axis=0)
    xy_boundary, step = np.linspace(lim0, lim1, n+1, retstep=True) # N x 2
    margin = .05 * step
    boundaries = []
    for ix in range(n):
        x0, x1 = xy_boundary[ix:ix+2,0]
        x0 -= margin[0]
        x1 += margin[0]
        for iy in range(n):
            y0, y1 = xy_boundary[iy:iy+2,1]
            y0 -= margin[1]
            y1 += margin[1]
            boundaries.append( (x0,x1,y0,y1) )
    pc_blocks = []
    sem_blocks = []
    ins_blocks = []
    for x0,x1,y0,y1 in boundaries:
        inblock = xyzrgb[:,0] > x0
        inblock = np.logical_and(inblock, xyzrgb[:,0] < x1)
        inblock = np.logical_and(inblock, xyzrgb[:,1] > y0)
        inblock = np.logical_and(inblock, xyzrgb[:,1] < y1)
        if not inblock.any():
            continue
        pc_sampled, sem_sampled, ins_sampled\
                = sample_data_label(xyzrgb[inblock,:], sem_points[inblock], ins_points[inblock], num_points)
        pc_blocks.append(pc_sampled)
        sem_blocks.append(sem_sampled)
        ins_blocks.append(ins_sampled)
    pc_blocks  = np.stack(pc_blocks, axis=0)
    sem_blocks = np.stack(sem_blocks,axis=0)
    ins_blocks = np.stack(ins_blocks,axis=0)
    return pc_blocks, sem_blocks, ins_blocks

def resize(rgb, depth, gt_marker, K):
    assert(False)
    osize = (depth.shape[1], depth.shape[0])
    dsize = (1280, 960)
    sK = K.copy()
    for i in range(2):
        sK[i,:] *= dsize[i] / osize[i]
    srgb       = cv2.resize(rgb, dsize, interpolation=cv2.INTER_NEAREST)
    sdepth     = cv2.resize(depth, dsize, interpolation=cv2.INTER_NEAREST)
    sgt_marker = cv2.resize(gt_marker, dsize, interpolation=cv2.INTER_NEAREST)
    return srgb, sdepth, sgt_marker, sK

class Data_ObbDataset:
    """
    @brief Replacement of Data_S3DIS for Data_ObbDataset.
    """

    def check_cache(self, name):
        cache_dir = osp.join(name,'cached_block')
        if osp.exists(cache_dir):
            return cache_dir
        makedirs(cache_dir)

        from unet.segment_dataset import ObbDataset
        from torch.utils.data import Dataset, DataLoader
        dataset = ObbDataset(name, False, max_frame_per_scene=1)
        # cache_dir, dataset
        indices = []
        sem_label_box = Data_Configs.sem_ids[ Data_Configs.sem_names.index('box') ]
        vnormalize = np.vectorize(xyz_normalize)

        for i, data in enumerate(dataset):
            print("cache %d/%d"%(i,len(dataset)) )
            bgr, depth, marker = data['rgb'], data['depth'], data['marker']
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            K = data['K']
            D = np.zeros((4,1),dtype=K.dtype)
            rgb, depth, marker, K = resize(rgb, depth, marker, K)
            xyzrgb, ins_points\
                    = UnprojectPointscloud(rgb,depth,marker,K,D,leaf_xy=0.02,leaf_z=0.01)
            if (xyzrgb[:,2]==0.).any():
                import pdb; pdb.set_trace()

            ins_points -= 1 # Data_S3DIS assign -1 for ins_labels of bg points.
            sem_points = np.full_like(ins_points, -1)
            sem_points[ins_points > -1] = sem_label_box

            xyzrgb = xyzrgb[ins_points>-1,:]
            ins_points = ins_points[ins_points > -1]
            sem_points = np.full_like(ins_points, sem_label_box)

            #in_range = xyzrgb[:,2] < 2. # TODO Parameter
            #xyzrgb     = xyzrgb[in_range]
            #ins_points = ins_points[in_range]
            #sem_points = sem_points[in_range]

            mins, maxs = np.amin(xyzrgb[:,:3],axis=0), np.amax(xyzrgb[:,:3],axis=0)
            wvec = maxs - mins
            for k in range(3):
                wvec[k] = max(wvec[k], 1e-5)
            normalized_xyz = np.zeros( (xyzrgb.shape[0],3), dtype=xyzrgb.dtype)
            for k in range(3):
                normalized_xyz[:,k] = vnormalize(xyzrgb[:,k], mins[k], wvec[k])
            xyzrgb = np.concatenate([xyzrgb, normalized_xyz], axis=-1)
            blocks = get_blocks(xyzrgb, ins_points, sem_points, num_points=Data_Configs.train_pts_num)
            # xyzrgb, sems, ins = blocks
            fn_frame = osp.join(cache_dir, '%d.pick'%data['idx'] )
            with open(fn_frame,'wb') as f_frame:
                data['blocks'] = blocks
                pickle.dump(data, f_frame, protocol=2)

            for j in range(blocks[0].shape[0]):
                indices.append( (i,j) )
        fn_info = osp.join(cache_dir, 'info.pick')
        with open(fn_info,'wb') as f:
            pickle.dump(indices, f, protocol=2)
        return cache_dir

    def __init__(self , name, batch_size=1, max_frame_per_scene=1):
        self.name = name
        cache_dir = self.check_cache(name)
        fn_info = osp.join(cache_dir, 'info.pick')
        self.scene_num = -1
        with open(fn_info,'rb') as f:
            indices = pickle.load(f)
        #indices = filter(lambda pair: pair[0]==0 , indices)

        for i,j in indices:
            self.scene_num = max(self.scene_num, i+1)
        self.cache_dir = cache_dir
        self.indices = indices
        self.batch_size = min(len(indices), batch_size)
        self.total_train_batch_num = int( len(indices) / self.batch_size )
        self.next_ch_index = 0
        self.next_scene_index = 0

    @staticmethod
    def get_bbvert_pmask_labels(pc, ins_labels):
        #from bonet.helper_data_s3dis import Data_S3DIS
        #return Data_S3DIS.get_bbvert_pmask_labels(pc, ins_labels)
        gt_bbvert_padded = np.zeros((Data_Configs.ins_max_num, 2, 3), dtype=np.float32)
        gt_pmask = np.zeros((Data_Configs.ins_max_num, pc.shape[0]), dtype=np.float32)
        count = -1
        unique_ins_labels = np.unique(ins_labels)
        for ins_ind in unique_ins_labels:
            if ins_ind <= -1: continue
            count += 1
            if count >= Data_Configs.ins_max_num:
                print('ignored! more than max instances:', len(unique_ins_labels));
                import pdb; pdb.set_trace()
                continue
        
            ins_labels_tp = np.zeros(ins_labels.shape, dtype=np.int8)
            ins_labels_tp[ins_labels == ins_ind] = 1
            ins_labels_tp = np.reshape(ins_labels_tp, [-1])
            gt_pmask[count,:] = ins_labels_tp
        
            ins_labels_tp_ind = np.argwhere(ins_labels_tp == 1)
            ins_labels_tp_ind = np.reshape(ins_labels_tp_ind, [-1])

            ###### bb min_xyz, max_xyz
            pc_xyz_tp = pc[:, 0:3]
            pc_xyz_tp = pc_xyz_tp[ins_labels_tp_ind]
            gt_bbvert_padded[count, 0, 0] = x_min = np.min(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 0, 1] = y_min = np.min(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 0, 2] = z_min = np.min(pc_xyz_tp[:, 2])
            gt_bbvert_padded[count, 1, 0] = x_max = max(np.max(pc_xyz_tp[:, 0]), x_min+1e-5)
            gt_bbvert_padded[count, 1, 1] = y_max = max(np.max(pc_xyz_tp[:, 1]), y_min+1e-5)
            gt_bbvert_padded[count, 1, 2] = z_max = max(np.max(pc_xyz_tp[:, 2]), z_min+1e-5)

        if False:
            print(unique_ins_labels)
            print('gt_pmask =', np.amax(gt_pmask,axis=1) )
            import pdb; pdb.set_trace() 
        return gt_bbvert_padded, gt_pmask


    def shuffle_train_files(self, ep):
        # ep : Have no idea that ep is short for what. but it is sued as random seed.
        # It shuffle self.train_files
        self.train_next_bat_index=0
        self.next_ch_index = 0
        self.next_scene_index = 0
        return

    def load_test_next_batch_random(self):
        return self.load_train_next_batch()

    def load_raw_data(self):
        if self.next_ch_index == len(self.indices):
            return None
        frame_idx, ch_idx = self.indices[self.next_ch_index]
        self.next_ch_index += 1
        if not hasattr(self, 'blocks') or self.frame_idx != frame_idx:
            fn_frame = osp.join(self.cache_dir, '%d.pick'% frame_idx)
            with open(fn_frame,'rb') as f_frame:
                if version_info.major == 3:
                    pick = pickle.load(f_frame, encoding='latin1')
                else:
                    pick = pickle.load(f_frame)
            self.pick = pick
            self.blocks = pick['blocks']
            self.frame_idx = frame_idx
        #print("file, ch/n(ch) = %d, %d/%d" % (frame_idx, ch_idx, self.blocks[0].shape[0]) )

        pc_xyzrgb = self.blocks[0][ch_idx,:,:]
        sems   = self.blocks[1][ch_idx,:]
        ins    = self.blocks[2][ch_idx,:]
        return self.pick, pc_xyzrgb, sems, ins

    def load_fixed_points(self):
        ret = self.load_raw_data()
        if ret is None:
            return None
        pick, pc_xyzrgb, sem_labels, ins_labels = ret
        if (pc_xyzrgb[:,2]<1e-5).any():
            print("Unexpected small depth")
            import pdb; pdb.set_trace()

        ### center xy within the block
        min_x = np.min(pc_xyzrgb[:,0]); max_x = np.max(pc_xyzrgb[:,0])
        min_y = np.min(pc_xyzrgb[:,1]); max_y = np.max(pc_xyzrgb[:,1])
        min_z = np.min(pc_xyzrgb[:,2]); max_z = np.max(pc_xyzrgb[:,2])
        ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
        pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x)/ np.maximum((max_x - min_x), 1e-3)
        pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y)/ np.maximum((max_y - min_y), 1e-3)
        pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z)/ np.maximum((max_z - min_z), 1e-3)
        pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)

        ########
        sem_labels = sem_labels.reshape([-1])
        ins_labels = ins_labels.reshape([-1])
        bbvert_padded_labels, pmask_padded_labels = self.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx]==-1: continue # invalid points
            sem_idx = Data_Configs.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] =1

        if np.isnan(bbvert_padded_labels).any():
            print("!!!!!! Nan in array")
            import pdb; pdb.set_trace()
        if np.isnan(pmask_padded_labels).any():
            print("!!!!!! Nan in array")
            import pdb; pdb.set_trace()

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels, pick

    def load_next_scene(self):
        bat_pc=[]
        bat_sem_labels=[]
        bat_ins_labels=[]
        bat_psem_onehot_labels =[]
        bat_bbvert_padded_labels=[]
        bat_pmask_padded_labels =[]
        while True:
            if self.next_ch_index == len(self.indices):
                frame_idx = self.indices[-1][0] + 1
            else:
                frame_idx, _ = self.indices[self.next_ch_index]
            if frame_idx != self.next_scene_index:
                break
            ret = self.load_fixed_points()
            if ret is None:
                break
            pc, sem_labels, ins_labels, psem_onehot_labels, \
                    bbvert_padded_labels, pmask_padded_labels, pick \
                    = ret
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)
        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)
        self.next_scene_index += 1
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels, pick

    def load_train_next_batch(self):
        bat_pc=[]
        bat_sem_labels=[]
        bat_ins_labels=[]
        bat_psem_onehot_labels =[]
        bat_bbvert_padded_labels=[]
        bat_pmask_padded_labels =[]
        #print('---------')
        for i in range(self.batch_size):
            #print('ch of batch : %d/%d, ch over all : %d/%d' % (i, self.batch_size, self.next_ch_index, len(self.indices)))
            ret = self.load_fixed_points()
            if ret is None:
                break
            pc, sem_labels, ins_labels, psem_onehot_labels, \
                    bbvert_padded_labels, pmask_padded_labels, _ = ret
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)
        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)
        if bat_pc.shape[0] == 0:
            import pdb; pdb.set_trace()
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels

class Data_VtkDataset(Data_ObbDataset):
    def check_cache(self, name):
        cache_dir = osp.join(name,'cached_block')
        if osp.exists(cache_dir):
            return cache_dir
        makedirs(cache_dir)
        #pick = { 'xyzrgb':xyzrgb, 'ins_points':ins_points, 'rgb':rgb, 'K':K, 'D':D}
        pick_files = glob2.glob(osp.join(name,'*.pick'),recursive=False)
        indices = []
        sem_label_box = Data_Configs.sem_ids[ Data_Configs.sem_names.index('box') ]
        vnormalize = np.vectorize(xyz_normalize)
        for i_pick, fn in enumerate(pick_files):
            img_idx = osp.basename(fn).split('.')[0]
            img_idx = int(img_idx)
            with open(fn,'rb') as f:
                if version_info.major == 3:
                    pick = pickle.load(f, encoding='latin1')
                else:
                    pick = pickle.load(f)
            xyzrgb, ins_points = pick['xyzrgb'], pick['ins_points']
            mins, maxs = np.amin(xyzrgb[:,:3],axis=0), np.amax(xyzrgb[:,:3],axis=0)
            wvec = maxs - mins
            for k in range(3):
                wvec[k] = max(wvec[k], 1e-5)
            normalized_xyz = np.zeros( (xyzrgb.shape[0],3), dtype=xyzrgb.dtype)
            for k in range(3):
                normalized_xyz[:,k] = vnormalize(xyzrgb[:,k], mins[k], wvec[k])
            xyzrgb = np.concatenate([xyzrgb, normalized_xyz], axis=-1)
            sem_points = np.full_like(ins_points, sem_label_box)

            ins_points -= 1 # Data_S3DIS assign -1 for ins_labels of bg points.
            sem_points[ins_points == -1] = -1

            if False:
                xyzrgb = xyzrgb[ins_points > -1]
                ins_points = ins_points[ins_points > -1]
                sem_points = sem_points[sem_points > -1]


            blocks = get_blocks(xyzrgb, ins_points, sem_points, num_points=Data_Configs.train_pts_num)
            fn_frame = osp.join(cache_dir, '%d.pick'%i_pick )
            with open(fn_frame,'wb') as f_frame:
                pick['blocks'] = blocks
                pickle.dump(pick, f_frame, protocol=2)
            for j in range(blocks[0].shape[0]):
                indices.append( (i_pick, j) )
        fn_info = osp.join(cache_dir, 'info.pick')
        with open(fn_info,'wb') as f:
            pickle.dump(indices, f, protocol=2)
        return cache_dir

