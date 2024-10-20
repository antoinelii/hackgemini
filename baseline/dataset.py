"""
Baseline Pytorch Dataset
"""

import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
import torch
import torch.utils

from baseline.collate import pad_collate

tile_mapping = {'t30uxv': 0.,
                't31tfj': 1.,
                't31tfm': 2.,
                't32ulu': 3.}

def parse_timestamps(patch_id, metadata):
    """
    Parse timestamps from metadata.
    Returns a 1D-tensor enhanced with cyclical date encoding
    in the following format : [year, month, day, cos_date, sin_date]
    """
    dates_dict = metadata["dates-S2"][patch_id]
    ymd_matrix = [[int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:8])] 
                  for x in dates_dict.values()]
    ymd_cyc_matrix = []
    for ymd in ymd_matrix:
        month = ymd[1]
        day = ymd[2]
        month_cos = np.cos(2 * np.pi * (month - 1 + day/31) / 12)
        month_sin = np.sin(2 * np.pi * (month - 1 + day/31) / 12)
        ymd_cyc_matrix.append(ymd + [month_cos, month_sin]) 
    return torch.tensor(ymd_cyc_matrix, dtype=torch.float32)


class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, folder: Path):
        super(BaselineDataset, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        print("Done.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        print("Dataset ready.")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        id_patch = self.id_patches[item]

        # Open and prepare satellite data into T x C x H x W arrays
        path_patch = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(path_patch).astype(np.float32)
        data = {"S2": torch.from_numpy(data)}
        
        # If you have other modalities, add them as fields of the `data` dict ...
        # data["radar"] = ...
        data["date"] = parse_timestamps(id_patch, self.meta_patch)
        tile_name = self.meta_patch["TILE"][id_patch]
        data["TILE"] = torch.tensor([tile_mapping[tile_name]])
        data["N_Parcel"] = torch.tensor([self.meta_patch["N_Parcel"][id_patch]])
        data["Parcel_Cover"] = torch.tensor([self.meta_patch["Parcel_Cover"][id_patch]])

        # Open and prepare targets
        target = np.load(
            os.path.join(self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch))
        )
        target = torch.from_numpy(target[0].astype(int))

        return data, target
    
class BaselineDatatest(torch.utils.data.Dataset):
    def __init__(self, folder: Path):
        super(BaselineDatatest, self).__init__()
        self.folder = folder

        # Get metadata
        print("Reading patch metadata ...")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        print("Done.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index
        print("Dataset ready.")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        id_patch = self.id_patches[item]

        # Open and prepare satellite data into T x C x H x W arrays
        path_patch = os.path.join(self.folder, "DATA_S2", "S2_{}.npy".format(id_patch))
        data = np.load(path_patch).astype(np.float32)
        data = {"S2": torch.from_numpy(data)}
        
        # If you have other modalities, add them as fields of the `data` dict ...
        # data["radar"] = ...
        data["date"] = parse_timestamps(id_patch, self.meta_patch)
        tile_name = self.meta_patch["TILE"][id_patch]
        data["TILE"] = torch.tensor([tile_mapping[tile_name]])
        data["N_Parcel"] = torch.tensor([self.meta_patch["N_Parcel"][id_patch]])
        data["Parcel_Cover"] = torch.tensor([self.meta_patch["Parcel_Cover"][id_patch]])

        return data
    

def get_train_val_Dataloaders(
        dt_train: torch.utils.data.Dataset,
        train_ratio: float=0.8,
        train_batch_size: int=8,
        val_batch_size: int=8,
        ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    # Set seeds for PyTorch
    torch.manual_seed(42)

    # Split dataset: train_ratio train, (1 - train_ratio) val
    train_size = int(train_ratio * len(dt_train))
    val_size = len(dt_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dt_train, [train_size, val_size])

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, collate_fn=pad_collate, shuffle=True
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, collate_fn=pad_collate, shuffle=False
        )

    return train_loader, val_loader

def get_test_Dataloader(
        dt_test: torch.utils.data.Dataset,
        test_batch_size: int=8
    ) -> torch.utils.data.DataLoader:
    test_loader = torch.utils.data.DataLoader(
        dt_test, batch_size=test_batch_size, collate_fn=pad_collate, shuffle=False
        )

    return test_loader