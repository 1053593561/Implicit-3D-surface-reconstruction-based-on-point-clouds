import os
import logging
import torch
from torch_geometric.data import Dataset, Data
import importlib
from pathlib import Path
import numpy as np
import trimesh

class SceneNet(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None, split="training", filter_name=None, dataset_size=None, 
                 point_density=None,
                 **kwargs):


        super().__init__(root, transform, None)

        logging.info("Dataset - SceneNet")

        self.split = split
        self.point_density = point_density

        self.filenames = [
        "CornerMap",
        "SurfMap",
        "GlobalMap",
        ]
        self.filenames = [root+filename for filename in self.filenames]
        self.filenames.sort()

        self.dataset_size = dataset_size
        if self.dataset_size is not None:
            self.filenames = self.filenames[:self.dataset_size]

        logging.info(f"Dataset - len {len(self.filenames)}")

    def _download(self): # override _download to remove makedirs
        pass

    def download(self):
        pass

    def _process(self):
        pass

    def len(self):
        return len(self.filenames)

    def get_category(self, idx):
        return self.filenames[idx].split("/")[-2]

    def get_object_name(self, idx):
        return self.filenames[idx].split("/")[-1]

    def get_class_name(self, idx):
        return "n/a"

    
    def get_data_for_evaluation(self, idx):
        raise NotImplementedError
        scene = self.filenames[idx]
        input_pointcloud = np.load(scene)
        return input_pointcloud, None


    def get(self, idx):
        """Get item."""

        # load the mesh
        scene_filename = self.filenames[idx]

        data = np.loadtxt(scene_filename+".xyz", dtype=np.float32)

        pos = data[:,:3]
        nls = data[:,3:]

        pos = torch.tensor(pos, dtype=torch.float)
        nls = torch.tensor(nls, dtype=torch.float)
        pos_non_manifold = torch.zeros((1,3), dtype=torch.float)


        data = Data(shape_id=idx, x=torch.ones_like(pos),
                    normal=nls,
                    pos=pos, pos_non_manifold=pos_non_manifold
                    )

        return data