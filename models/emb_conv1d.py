"""
This is method 3 where we applied conv1d for the output from two Convnext networks
"""

import torch
import torch.nn as nn

from .resnet import resnet50
from .convnext import convnext_T
from .regressor import Regressor

class emb_conv1d(nn.Module):
    def __init__(self, smpl_mean_params, pretrained=True):
        super(emb_conv1d, self).__init__()
        self.encoder = convnext_T(pretrained=pretrained)
        self.regressor = Regressor(smpl_mean_params, encoder = 'convnext')
        # resnet embedding_size = 2048
        # convnext embedding_size = 1000
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        )
        print('### Hello Conv1d ###')

    def forward(self, img0,img1, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        #resnet embeddings
        e1 = self.encoder.forward(img1)#compute e1 first to cache forward info of e0
        e0 = self.encoder.forward(img0)
        # print('e1.shape', e1.shape) # should be (batch_size, 2048)

        # Obtain the initial pose parameters
        pred_rotmat0, pred_betas0, pred_camera0=self.regressor.forward(e0, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, n_iter=3)
        

        # pelvis and camera of img0 SMPL param
        pelvis0 = pred_rotmat0[:,[0]]
        camera0 = pred_camera0

        e_stacked = torch.stack((e0,e1), dim=1)
        e_combined = self.conv1d.forward(e_stacked)
        e_combined = e_combined.view(e_combined.shape[0], -1) # shape should be (batch_size, 1000) if convnext

        pred_rotmat_avg, pred_betas_avg, pred_camera_avg=self.regressor.forward(e_combined, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, n_iter=3)

        #pelvis of img0, rest of pos of avged, beta of avged, camera of img0
        pred_rotmat_final=torch.cat((pelvis0,pred_rotmat0[:,1:]),dim=1)
        pred_betas_final=pred_betas_avg
        pred_camera_final=pred_camera0

        return pred_rotmat_final,pred_betas_final,pred_camera_final
