"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
"""

import torch
import math
from scipy.spatial.transform import Rotation as R

import gc
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import hmr, SMPL
from datasets.base_dataset_multiview import BaseDataset
#from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
#from utils.part_utils import PartRenderer

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='h36m-p1', choices=['h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp'], help='Choose evaluation dataset')
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

def cam_to_world(loc3D, R, T):
    """ Convert local 3D points to global coordinate
    """
    print('T shape is: ', T.shape)
    to_return = []
    for i in range(loc3D.shape[0]):
        print(loc3D[i, :, :].shape)

        print(R[i, :])
        print((loc3D[i, :, :] - T[i, :]).shape)

        glob3D = R[i, :].T.dot((loc3D[i, :, :] - T[i, :]).T)
        to_return.append(glob3D.T)

    return to_return

def Extract_Angle(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular: 
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    #return x*180/np.pi, y*180/np.pi, z*180/np.pi
    return x, y, z


def run_evaluation(model, dataset_name, dataset, result_file,
                   batch_size=32, img_res=224, 
                   num_workers=24, shuffle=False, log_freq=50):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    
    #renderer = PartRenderer()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
    # Create dataloader for the dataset
    print('dataset length: ', len(dataset))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #data_loader = DataLoader(dataset, batch_size=2, shuffle=shuffle, num_workers=1)
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    mpjpe_avg = np.zeros(len(dataset))
    recon_err_avg = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    # Choose appropriate evaluation for each dataset
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True

    joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    #print('dataset length:', len(data_loader))
    # print("###################2############")
    # print('dataset 2 length: ', len(dataset))
    # print(dataset[130]['imgname0'])
    # print(dataset[130]['imgname1'])
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        torch.cuda.empty_cache()
        gc.collect()

        # 
        images_0 = batch['img0'].to(device)
        images_1 = batch['img1'].to(device)

        gt_pose0 = batch['pose0'].to(device)
        gt_pose1 = batch['pose1'].to(device)
        
        gt_betas0 = batch['betas0'].to(device)
        gt_betas1 = batch['betas1'].to(device)


        gt_vertices0 = smpl_neutral(betas=gt_betas0, body_pose=gt_pose0[:, 3:], global_orient=gt_pose0[:, :3]).vertices
        gt_vertices1 = smpl_neutral(betas=gt_betas1, body_pose=gt_pose1[:, 3:], global_orient=gt_pose1[:, :3]).vertices
        
        gender0 = batch['gender0'].to(device)
        gender1 = batch['gender1'].to(device)

        curr_batch_size0 = images_0.shape[0]
        curr_batch_size1 = images_1.shape[1]


        # # Get ground truth annotations from the batch
        # gt_pose = batch['pose'].to(device)
        # gt_betas = batch['betas'].to(device)
        # gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        # images = batch['img'].to(device)
        # gender = batch['gender'].to(device)
        # curr_batch_size = images.shape[0]

    #     print('pass 1')
        
        with torch.no_grad():
            pred_rotmat0, pred_betas0, pred_camera0 = model(images_0)
            pred_output0 = smpl_neutral(betas=pred_betas0, body_pose=pred_rotmat0[:,1:], global_orient=pred_rotmat0[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices0 = pred_output0.vertices

            pred_rotmat1, pred_betas1, pred_camera1 = model(images_1)
            pred_output1 = smpl_neutral(betas=pred_betas1, body_pose=pred_rotmat1[:,1:], global_orient=pred_rotmat1[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices1 = pred_output1.vertices

            
       # print(pred_rotmat0.shape)

        def average_angle(rot0, rot1):
            to_return = np.zeros_like(rot0)
           # print(to_return.shape)
            for batch in range(rot0.shape[0]):
                joints0 = rot0[batch]
                joints1 = rot1[batch]

                to_return[batch, 0, :, :] = joints0[0, :, :]

                for joint in range(1, 24):
                    #x0, y0, z0, = Extract_Angle(joints0[joint])
                    #x1, y1, z1, = Extract_Angle(joints1[joint])
                    #r = R.from_matrix(joints0[joint]).as_rotvec()
                    #print(r)

                    r0 = R.from_matrix(joints0[joint])
                    r1 = R.from_matrix(joints1[joint])

                    angles0 = r0.as_euler('xyz', degrees=True)
                    angles1 = r1.as_euler('xyz', degrees=True)

                    avg_angles = np.mean([angles0, angles1], axis=0)

                    avg_rot = R.from_euler('xyz', avg_angles, degrees=True)
                    avg_matrix = avg_rot.as_matrix()

                    to_return[batch, joint, :, :] = avg_matrix

            return np.array(to_return)


        #print(type(pred_rotmat0))
        avg_rotmat = average_angle(pred_rotmat0.cpu().numpy(), pred_rotmat1.cuda().cpu().numpy())
        pred_avg_rotmat = torch.from_numpy(avg_rotmat).to(device)
        pred_output_avg = smpl_neutral(betas=pred_betas0, body_pose=pred_avg_rotmat[:,1:], global_orient=pred_avg_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices_avg = pred_output_avg.vertices
            

        # if save_results:
        #     rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
        #     rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
        #     pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
        #     smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
        #     smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
        #     smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
            
    
        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices0.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name:
                gt_keypoints_3d_0 = batch['pose_3d0'].cuda()
                gt_keypoints_3d_0 = gt_keypoints_3d_0[:, joint_mapper_gt, :-1]

                gt_keypoints_3d_1 = batch['pose_3d1'].cuda()
                gt_keypoints_3d_1 = gt_keypoints_3d_1[:, joint_mapper_gt, :-1]
                #gt_keypoints_3d_0 = batch['keypoints3d0'].cuda()
                #gt_keypoints_3d_1 = batch['keypoints3d1'].cuda()
                
                # R0 = batch['R0'].to(device)
                # R1 = batch['R1'].to(device)

                # T0 = batch['T0'].to(device)
                # T1 = batch['T1'].to(device)

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d_0 = torch.matmul(J_regressor_batch, pred_vertices0)
            pred_keypoints_3d_1 = torch.matmul(J_regressor_batch, pred_vertices1)
            pred_keypoints_3d_avg = torch.matmul(J_regressor_batch, pred_vertices_avg)
            
            # if save_results:
            #     pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis_0 = pred_keypoints_3d_0[:, [0],:].clone()
            pred_keypoints_3d_0 = pred_keypoints_3d_0[:, joint_mapper_h36m, :] 
            pred_keypoints_3d_0 = pred_keypoints_3d_0 - pred_pelvis_0

            gt_keypoints_3d_0 = gt_keypoints_3d_0.squeeze()
            # gt_keypoints_3d_0 = gt_keypoints_3d_0 / 1000
            #gt_keypoints_3d_0 = gt_keypoints_3d_0 - gt_keypoints_3d_0[4] # 4 is the root

            pred_pelvis_1 = pred_keypoints_3d_1[:, [0],:].clone()
            pred_keypoints_3d_1 = pred_keypoints_3d_1[:, joint_mapper_h36m, :] 
            pred_keypoints_3d_1 = pred_keypoints_3d_1 - pred_pelvis_1

            #
            gt_keypoints_3d_1 = gt_keypoints_3d_1.squeeze()
            # gt_keypoints_3d_0 = gt_keypoints_3d_0 / 1000
            #gt_keypoints_3d_1 = gt_keypoints_3d_1 - gt_keypoints_3d_1[4] # 4 is the root
            #print(pred_keypoints_3d_0.shape)

            #pred_keypoints_3d_0 = cam_to_world(pred_keypoints_3d_0, R0, T0)
            #pred_keypoints_3d_1 = cam_to_world(pred_keypoints_3d_1, R1, T1)

            # print('predict', pred_keypoints_3d_0)
            # print('gt', gt_keypoints_3d_0)

            pred_pelvis_avg = pred_keypoints_3d_avg[:, [0],:].clone()
            pred_keypoints_3d_avg = pred_keypoints_3d_avg[:, joint_mapper_h36m, :] 
            pred_keypoints_3d_avg = pred_keypoints_3d_avg - pred_pelvis_avg

            

            #print('gt key points 3d ', gt_keypoints_3d_0)
            #print('pred key points 3d ', pred_keypoints_3d_0)

            # Absolute error (MPJPE)
            error0 = torch.sqrt(((pred_keypoints_3d_0 - gt_keypoints_3d_0) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            error1 = torch.sqrt(((pred_keypoints_3d_1 - gt_keypoints_3d_1) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            error_avg = torch.sqrt(((pred_keypoints_3d_avg - gt_keypoints_3d_0) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            
            #error = error0
            error = (error0 + error1) / 2
            error = error0
            
            mpjpe[step * batch_size:step * batch_size + curr_batch_size0] = error
            mpjpe_avg[step * batch_size:step * batch_size + curr_batch_size0] = error_avg

            # Reconstuction_error
            r_error0 = reconstruction_error(pred_keypoints_3d_0.cpu().numpy(), gt_keypoints_3d_0.cpu().numpy(), reduction=None)
            r_error1 = reconstruction_error(pred_keypoints_3d_1.cpu().numpy(), gt_keypoints_3d_1.cpu().numpy(), reduction=None)
            r_error_avg = reconstruction_error(pred_keypoints_3d_avg.cpu().numpy(), gt_keypoints_3d_0.cpu().numpy(), reduction=None)
            
            r_error = (r_error0 + r_error1) /2 
            r_error = r_error0
            
            recon_err[step * batch_size:step * batch_size + curr_batch_size0] = r_error
            recon_err_avg[step * batch_size:step * batch_size + curr_batch_size0] = r_error_avg

            # print("error: ", error)
            # print('r_error: ', r_error)

            # exit()

        # If mask or part evaluation, render the mask and part images
        #if eval_masks or eval_parts:
            #mask, parts = renderer(pred_vertices, pred_camera)

        # Print intermediate results during evaluation

        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print()

                print('MPJPE avg: ' + str(1000 * mpjpe_avg[:step * batch_size].mean()))
                print('Reconstruction Error avg: ' + str(1000 * recon_err_avg[:step * batch_size].mean()))
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        np.savez(result_file, pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)
    # Print final results during evaluation
    print('*** Final Results ***')
    print()
    if eval_pose:
        print('MPJPE: ' + str(1000 * mpjpe.mean()))
        print('Reconstruction Error: ' + str(1000 * recon_err.mean()))
        print()

        print('MPJPE avg: ' + str(1000 * mpjpe_avg.mean()))
        print('Reconstruction Error avg: ' + str(1000 * recon_err_avg.mean()))
        print()

if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # Setup evaluation dataset
    dataset = BaseDataset(None, args.dataset, is_train=False)
    print("###############################")
    print('dataset 2 length: ', len(dataset))
    # print(dataset[130]['imgname0'])
    # print(dataset[130]['imgname1'])
    # for i in range(100):
    #     print(dataset.imgname[i])

    # Run evaluation
    run_evaluation(model, args.dataset, dataset, args.result_file,
                   batch_size=args.batch_size,
                   shuffle=args.shuffle,
                   log_freq=args.log_freq)
