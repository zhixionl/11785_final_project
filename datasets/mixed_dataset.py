"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .base_dataset_multiview import BaseDataset
#from .base_dataset import BaseDataset  # Change to this if you want to use monocular 

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self, options, **kwargs):
        self.dataset_list = ['mpi-inf-3dhp']

        self.dataset_dict = {'mpi-inf-3dhp': 5}
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        self.length = max([len(ds) for ds in self.datasets])

        """
        Data distribution inside each batch:
        0% H36M - 100% MPI-INF
        """

        self.partition = [1.0]
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(6):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]
            

    def __len__(self):
        return self.length
