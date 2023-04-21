"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join
import os

H36M_ROOT = ''
LSP_ROOT = ''
LSP_ORIGINAL_ROOT = ''
LSPET_ROOT = ''
MPII_ROOT = ''
COCO_ROOT = ''
MPI_INF_3DHP_ROOT = '..\mpi_inf_3dhp'
PW3D_ROOT = ''
UPI_S1H_ROOT = ''

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data\dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'data\openpose'


# Path to test/train npz files
DATASET_FILES = [ {'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                  },

                  {
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz')
                  }
                ]

DATASET_FOLDERS = {
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
