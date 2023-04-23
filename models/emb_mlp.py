import torch
import torch.nn as nn
from .resnet import resnet50
from .regressor import Regressor


class emb_mlp(nn.Module):
    def __init__(self, smpl_mean_params, pretrained=True):
        super(emb_mlp, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)
        self.regressor = Regressor(smpl_mean_params)
        self.mlp=nn.Sequential(
            torch.nn.Linear(4096, 4096),
            torch.nn.GELU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.GELU(),
            torch.nn.Linear(2048, 2048)
        )

    def forward(self, img0,img1, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        #resnet embeddings
        e1=self.resnet.forward(img1)#compute e1 first to cache forward info of e0
        e0=self.resnet.forward(img0)

        #
        pred_rotmat0, pred_betas0, pred_camera0=self.regressor.forward(e0, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, n_iter=3)
        

        #pelvis and camera of img0 SMPL param
        pelvis0=pred_rotmat0[:,0]
        camera0=pred_camera0

        # pose0_no_pelvis=pred_rotmat0[:,1:]
        # shape0=pred_betas0

        # averaged_embedding=(e0+e1)/2
        e_concat=torch.cat((e0,e1),dim=1)
        e_combined=self.mlp.forward(e_concat)

        pred_rotmat_avg, pred_betas_avg, pred_camera_avg=self.regressor.forward(e_combined, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, n_iter=3)

        #pelvis of img0, rest of pos of avged, beta of avged, camera of img0
        pred_rotmat_final=torch.cat((pelvis0,pred_rotmat0[:,1:]),dim=1)
        pred_betas_final=pred_betas_avg
        pred_camera_final=pred_camera0

        return pred_rotmat_final,pred_betas_final,pred_camera_final
