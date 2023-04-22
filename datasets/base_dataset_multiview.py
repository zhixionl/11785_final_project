from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
import random 
import os
from os.path import join

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        # Change 
        #self.data = np.load(config.DATASET_FILES[True][dataset])
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.data=dict(self.data)

        self.imgname = self.data['imgname']

        # get index pairs of same frame, different video(view)
        if dataset == 'mpi-inf-3dhp':
            indices = np.empty((8,2), dtype=object)
            self.pairs = []
            self.Ks = []
            self.Rs = []
            self.Ts = []

            for idx, img in enumerate(self.imgname):
               # print(img)
                img = img.split('/')
                #print(img)
                S = int(img[0][-1])
                # print('S', S)
                Seq = int(img[1][-1])
                # print('Seq', Seq)
                Video = int(img[3][-1])
                # print('Video', Video)
                Frame = int(img[4][-10:-4])
                # print('Frame', Frame)

                if indices[S-1, Seq-1] is None:
                    indices[S-1, Seq-1] = {}
                if Frame not in indices[S-1, Seq-1]:
                    indices[S-1, Seq-1][Frame] = []
                indices[S-1, Seq-1][Frame].append(idx)
                # break
            random.seed(10)
            for S in range(8):
                if S == 7:
                    for Seq in range(2):
                        for key in indices[S,Seq]:
                            if len(indices[S, Seq][key]) >= 2:
                                #self.pairs.append(indices[S, Seq][key])
                                ordering = random.sample(range(len(indices[S, Seq][key])), len(indices[S, Seq][key]))
                                for i in range(len((indices[S, Seq][key]))// 2):
                                    self.pairs.append((indices[S, Seq][key][ordering[2*i]], 
                                                        indices[S, Seq][key][ordering[2*i+1]]))
                                

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        # self.keypoints2d = self.data['keypoints_2d']
        # self.keypoints3d = self.data['keypoints_3d']
        # self.keypoints3d_uni = self.data['keypoints_3d_uni']
        # self.Rs = self.data['R']
        # self.Ts = self.data['T']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        #self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def get_item_original(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])

        orig_shape = None
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
            orig_shape = np.array(img.shape)[:2]
        except TypeError:
            print('could not find'+imgname)


        # Get SMPL parameters, if available
        # if self.has_smpl[index]:
        #     pose = self.pose[index].copy()
        #     betas = self.betas[index].copy()
        # else:
        pose = np.zeros(72)
        betas = np.zeros(10)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['scaled_3d_keypts'] = S 
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)
            
        #item['keypoints2d'] = self.keypoints2d[index]
        #item['keypoints3d'] = self.keypoints3d[index]
        #item['keypoints3d_uni'] = self.keypoints3d_uni[index]

        # Get 2D keypoints and apply augmentation transforms
        #keypoints = self.keypoints[index].copy()
        #item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()

        #item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
      #  item['R'] = self.Rs[index]
      #  item['T'] = self.Ts[index]

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item
    
    def __getitem__(self, index):
        items = []
        temp = {}
        #print('pair')
        #print("22222222s: ", self.pairs[index])
        for i,p in enumerate(self.pairs[index]):
            #print('1111111111111111111111s',len(self.pairs[index]))
            #print(self.pairs[index])
            
            item = self.get_item_original(p)
            #img_name = item['imgname']
            # K, R, T = self.read_calibration(img_name)
            # # print(img_name)
            # # print(K)
            # # print(R)
            # # print(T)
            # item['K'] = K
            # item['R'] = R
            # item['T'] = T

            # calcualte projected 3D global keypts 
            
            # de-normalize S as 3D keypoints
            #print("current img_name ", img_name)
           # print(item['scaled_3d_keypts'][:, :3])
            #keypts_3d_global = self.cam_to_world(item['scaled_3d_keypts'][:, :3], R, T)
            #keypts_3d_global = item['keypoints3d_uni']
            #print('global ', keypts_3d_global)
      
            for k in item:
                temp[k+str(i)] = item[k]
            
        return temp

    def __len__(self):
        return len(self.pairs)
    
    # method for calibration file 
    def read_calibration(self, imgname):
        img = imgname.split('/')
        S = img[2]
        Seq = img[3]
        video = img[5]
        video_num = int(video[-1])


        vid_file = os.path.join(config.MPI_INF_3DHP_ROOT, S, Seq, 'camera.calibration')

        file = open(vid_file, 'r')
        content = file.readlines()
        vid_i = video_num
        
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]

        file.close()
        return K, R, T
    

    def cam_to_world(self,local3D, R, T):
        # convert the local 3D keypoints from cameras view to global view
        global3D = R.T.dot(local3D.T - T[:, None])

        return global3D.T
    

                            #     print(self.data['imgname'][indices[S, Seq][key]])
                            # video_name = video_num = self.data['imgname'][indices[S, Seq][key]].split("/")[2]
                            # print(video_name)

                            # cam_file = os.path.join(config.MPI_INF_3DHP_ROOT, S, Seq, 'camera.calibration')
                            # #x, y, z = read_calibration(cam_file, )