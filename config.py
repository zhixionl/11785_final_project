"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = ''
MPI_INF_3DHP_ROOT = '/home/jack/Desktop/Project/SPIN-master/datasets/MPI_INF_3DHP'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = '/home/jack/Desktop/Project/SPIN-master/data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = '/home/jack/Desktop/Project/SPIN-master/datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test_s8.npz'),
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train_s1-7.npz')
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                }

CUBE_PARTS_FILE = '/home/jack/Desktop/Project/SPIN-master/data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = '/home/jack/Desktop/Project/SPIN-master/data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = '/home/jack/Desktop/Project/SPIN-master/data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = '/home/jack/Desktop/Project/SPIN-master/data/vertex_texture.npy'
STATIC_FITS_DIR = '/home/jack/Desktop/Project/SPIN-master/data/static_fits'
SMPL_MEAN_PARAMS = '/home/jack/Desktop/Project/SPIN-master/data/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/home/jack/Desktop/Project/SPIN-master/data/smpl'
