#!/usr/bin/python
"""
Preprocess datasets and generate npz files to be used for training testing.
It is recommended to first read datasets/preprocess/README.md
"""
import argparse
import config as cfg
from datasets.preprocess.mpi_inf_3dhp_preprocess import mpi_inf_3dhp_extract
#from datasets.preprocess.mpi_inf_3dhp_preprocess_video import mpi_inf_3dhp_extract  # only for creating the videos

parser = argparse.ArgumentParser()
parser.add_argument('--train_files', default=False, action='store_true', help='Extract files needed for training')
parser.add_argument('--eval_files', default=False, action='store_true', help='Extract files needed for evaluation')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # define path to store extra files
    out_path = cfg.DATASET_NPZ_PATH
    openpose_path = cfg.OPENPOSE_PATH

    # if args.train_files:
        # MPI-INF-3DHP dataset preprocessing (training set)
    mpi_inf_3dhp_extract(cfg.MPI_INF_3DHP_ROOT, openpose_path, out_path, 'train', extract_img=True, static_fits=cfg.STATIC_FITS_DIR)