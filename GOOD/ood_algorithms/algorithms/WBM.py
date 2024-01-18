"""
Base class for OOD algorithms
"""
from abc import ABC
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg

@register.ood_alg_register
class WBM(BaseOODAlg):
    r"""
        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(WBM, self).__init__(config)
        self.optimizer: torch.optim.Adam = None
        self.scheduler: torch.optim.lr_scheduler._LRScheduler = None
        self.model: torch.nn.Module = None

        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0


    def set_up(self, model: torch.nn.Module, loader, config: Union[CommonArgs, Munch]):

        self.model: torch.nn.Module = model

        X_train = []
        Y_train = []
        F_train = []
        #data is a Batch
        for data in loader['train']:
            ptr = data.ptr
            coo = to_scipy_sparse_matrix(data.edge_index)
            indices = np.vstack((coo.row, coo.col))
            values = coo.data
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape
            C_batch = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
            for j in range(data.y.shape[0]):
                X_train.append(C_batch[ptr[j]:ptr[j+1], ptr[j]:ptr[j+1]])
                Y_train.append(data.y[j].to(torch.long))
                F_train.append(data.x[ptr[j]:ptr[j+1], :])

        shapes = [C.shape[0] for C in X_train]
        if config.dataset.dataset_name == "GOODMotif" or config.dataset.dataset_name == "GOODHIV":
            Satoms = round(np.median(shapes))
        else:
            Satoms = np.max(shapes)

        self.model.init_parameters_with_aggregation(
            Satoms=Satoms, init_mode_atoms='sampling_supervised_median',
            labels=torch.tensor(Y_train, device=config.device, dtype=torch.long),
            graphs=X_train, features=F_train, atoms_projection='clipped')


        self.optimizer = torch.optim.Adam(self.model.params, lr=config.train.lr, betas=(0.9, 0.99),
                                          weight_decay=config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.train.mile_stones,
                                                              gamma=0.1)


    def backward(self, loss):
        r"""
        Gradient backward process and parameter update.

        Args:
            loss: target loss
        """
        loss.backward()
        self.optimizer.step()
        self.model.prune_parameters()
