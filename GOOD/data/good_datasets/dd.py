import math
import os
import os.path as osp
from os.path import join as opj
import torch
from munch import Munch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.datasets import TUDataset as _TUDataset

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *

def size_split(dataset: Dataset, split_path) -> (Dataset, Dataset, Dataset):
    train_idxes = np.loadtxt(opj(split_path, f"train_idx.txt"), dtype=np.int64)
    val_idxes = np.loadtxt(opj(split_path, f"val_idx.txt"), dtype=np.int64)
    test_idxes = np.loadtxt(opj(split_path, f"test_idx.txt"), dtype=np.int64)
    return (
        dataset[torch.from_numpy(train_idxes)],
        dataset[torch.from_numpy(val_idxes)],
        dataset[torch.from_numpy(test_idxes)],
    )

class TUDataset(_TUDataset):
    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, name)  # osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, name)  # osp.join(self.root, self.name, name)

def load_TUDatasets(dataset_root) -> (Dataset, Dataset, Dataset):
    dataset_path = opj(dataset_root, "DD", str(0))
    ret = TUDataset(
        opj(dataset_path, "original"),
        name="DD",
        use_node_attr=False,
        pre_transform=None,
        transform=None
    )

    train, val, test = size_split(ret, opj(dataset_root, "DD"))
    return train, val, test

@register.dataset_register
class DD(InMemoryDataset):

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.metric = 'MCC'
        self.task = 'Unbalanced classification'
        super().__init__(root, transform, pre_transform)


    @staticmethod
    def load(dataset_root: str, domain=None, shift=None, generate=None):
        meta_info = Munch()
        meta_info.dataset_type = 'syn'
        meta_info.model_level = 'graph'

        train_dataset, val_dataset, test_dataset = load_TUDatasets(dataset_root)


        meta_info.dim_node = 89
        meta_info.dim_edge = 0

        meta_info.num_envs = 2
        meta_info.num_classes = 2


        # --- clear buffer dataset._data_list ---
        train_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': None, 'id_test': None,
                'val': val_dataset, 'test': test_dataset, 'task': 'Unbalanced Binary Classification',
                'metric': 'MCC'}, meta_info
