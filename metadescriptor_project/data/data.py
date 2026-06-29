from sklearn.model_selection import StratifiedKFold, train_test_split
import lightning as L
import torch
import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from typing import Optional
from torch_geometric.datasets import TUDataset
from pathlib import Path

class KfoldGraphDataModule(L.LightningDataModule):
    def __init__(
        self,
        name: str = "AIDS",
        data_dir: str = "data/TUDataset/TUDataset",
        k: int = 1,  # fold number
        num_splits: int = 10,
        batch_size: int = 64,
    ):
        super().__init__()
        self.name = (name,)
        self.k = k
        self.num_splits = num_splits
        self.input_dim = None
        self.batch_size = batch_size
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.test = None
        self.weights=None
        assert 0 <= self.k <= self.num_splits, "incorrect fold number"
        self.save_hyperparameters()

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            dataset_full = TUDataset(
                root= Path.cwd().joinpath(self.hparams.data_dir),
                name=self.hparams.name,
                use_node_attr=True,
                use_edge_attr=True,
            )
            sample_graph = dataset_full[0]  # Premier graphe
            self.input_dim = sample_graph.x.size(1)
            # left 20% of the data for the test
            train_idx, test_idx = train_test_split(
            np.arange(len(dataset_full)),
            test_size=0.2,           
            stratify=  dataset_full.y          
        )
        
            self.test = dataset_full[test_idx]    
            self.data_cv = dataset_full[train_idx]
            kf = StratifiedKFold(
                n_splits=self.hparams.num_splits,
                shuffle=True,
            )
            all_splits = [k for k in kf.split(self.data_cv,self.data_cv.y )]
            train_indexes, val_indexes = all_splits[self.hparams.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            self.data_train, self.data_val = (
                self.data_cv[train_indexes],
                self.data_cv[val_indexes],
            )
            class_counts = torch.bincount(dataset_full.y)
            total_samples = len(dataset_full)
            
            weights = total_samples / (len(class_counts) * class_counts.float())
            self.weights =  weights

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,pin_memory=True,num_workers=4, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(dataset=  self.test, batch_size=self.batch_size,pin_memory=True,num_workers=4)
