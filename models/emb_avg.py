"""
This is method 2 where we averaged the features from multi-view images with default
RESNET 50 network
"""

import torch
import torch.nn as nn
from .resnet import resnet50
from .regressor import Regressor

class emb_avg(nn.Module):
    def __init__(self, smpl_mean_params, pretrained=True):
        super(emb_avg, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)
        self.regressor = Regressor(smpl_mean_params, encoder='resnet')
        print("### Welcome to Method 2 ###")

    def forward(self, img0,img1, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        # Resnet embeddings
        e1=self.resnet.forward(img1)#compute e1 first to cache forward info of e0
        e0=self.resnet.forward(img0)

        # Obtain the initial pose parameters
        pred_rotmat0, pred_betas0, pred_camera0=self.regressor.forward(e0, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, n_iter=3)
        
        #pelvis and camera of img0 SMPL param
        pelvis0 = pred_rotmat0[:,[0]]
        camera0 = pred_camera0

        averaged_embedding=(e0+e1)/2

        pred_rotmat_avg, pred_betas_avg, pred_camera_avg=self.regressor.forward(averaged_embedding, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, n_iter=3)

        # pelvis of img0, rest of pos of avged, beta of avged, camera of img0
        pred_rotmat_final=torch.cat((pelvis0,pred_rotmat_avg[:,1:]),dim=1)
        pred_betas_final=pred_betas_avg
        pred_camera_final=pred_camera0

        return pred_rotmat_final,pred_betas_final,pred_camera_final