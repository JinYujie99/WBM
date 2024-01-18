import copy
import math

import torch
import ot
import time
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree, to_scipy_sparse_matrix

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.GW_utils import probability_simplex_projection
from GOOD.utils.utils_networks import MLP, MLP_dropout
from sklearn.cluster import KMeans
from .BaseGNN import GNNBasic
from .GINvirtualnode import vGINFeatExtractor

@register.model_register
class vGIN_WBM(GNNBasic):
    def __init__(self, config: Union[CommonArgs, Munch], deg):
        super().__init__(config)

        self.Katoms = config.model.Katoms
        self.num_classes = config.dataset.num_classes
        self.config = config
        self.dtype = torch.float64
        self.device = config.device
        self.Cbar, self.Fbar, self.hbar = None, None, None

        input_dim = self.Katoms

        self.gnn = vGINFeatExtractor(config)
        if config.model.clf_dropout != 0.:
            self.classifier = MLP_dropout(
                input_dim = input_dim,
                output_dim = self.num_classes,
                hidden_dim = config.model.clf_hidden_dim,
                num_hidden = config.model.clf_num_hidden,
                output_activation= 'linear',
                dropout = config.model.clf_dropout,
                skip_first = True,
                device=self.device, dtype=self.dtype)
        else:
            self.classifier = MLP(
                input_dim = input_dim,
                output_dim = self.num_classes,
                hidden_dim = config.model.clf_hidden_dim,
                num_hidden = config.model.clf_num_hidden,
                output_activation='linear', device=self.device, dtype=self.dtype)

        self.learn_hbar = config.model.learn_hbar


    def init_parameters_with_aggregation(self,
                                         Satoms: int,
                                         init_mode_atoms: str,
                                         graphs: list = None, features: list = None, edges: list = None,
                                         labels: list = None, atoms_projection: str = 'clipped'):
        list_Satoms = [Satoms] * self.Katoms

        self.Cbar = []
        self.Fbar = []
        self.hbar = []
        self.ebar = []
        for S in list_Satoms:
            x = torch.ones(S, dtype=self.dtype, device=self.device) / S
            x.requires_grad_(self.learn_hbar)
            self.hbar.append(x)

        if 'sampling_supervised' in init_mode_atoms:
            print('init_mode_atoms = sampling_supervised')
            # If not enough samples have the required shape within a label
            # we get the samples within the label and create perturbated versions of observed graphs
            # Then we do a forward pass within the GIN networks to get embedded features
            # of sampled graphs.
            shapes = torch.tensor([C.shape[0] for C in graphs])
            unique_atom_shapes, counts = np.unique(list_Satoms, return_counts=True)
            unique_labels = torch.unique(labels)
            idx_by_labels = [torch.where(labels == label)[0] for label in unique_labels]

            for i, shape in enumerate(unique_atom_shapes):
                r = counts[i]
                perm = torch.randperm(unique_labels.shape[0])
                count_by_label = r // unique_labels.shape[0]
                for i in perm:
                    i_ = i.item()
                    shapes_label = shapes[idx_by_labels[i_]]
                    shape_idx_by_label = torch.where(shapes_label == shape)[0]
                    print('shape_idx_by_label (S=%s) / %s' % (shape, shape_idx_by_label))
                    if shape_idx_by_label.shape[0] == 0:
                        print('no samples for shape S=%s -- computing kmeans on features within the label' % shape)
                        stacked_features = torch.cat(features)
                        print('stacked features shape:', stacked_features.shape)
                        km = KMeans(n_clusters=shape, init='k-means++', n_init=10, random_state=0)
                        km.fit(stacked_features)

                        F_clusters = torch.tensor(km.cluster_centers_, dtype=self.dtype, device=self.device)
                        print('features from clusters:', F_clusters.shape)
                    try:
                        print('found samples of shape =%s / within label  =%s' % (shape, i_))
                        sample_idx = np.random.choice(shape_idx_by_label.numpy(), size=min(r, count_by_label),
                                                      replace=False)
                        print('sample_idx:', sample_idx)
                        for idx in sample_idx:
                            localC = graphs[idx_by_labels[i_][idx]].clone().to(self.device)
                            localC.to(self.dtype)
                            localC.requires_grad_(True)
                            localF = features[idx_by_labels[i_][idx]].clone().to(self.device)
                            # localF.to(self.dtype)
                            localE = edges[idx_by_labels[i_][idx]].clone().to(self.device)
                            localE.to(self.dtype)
                            # localF.requires_grad_(True)
                            self.Cbar.append(localC)
                            self.Fbar.append(localF)
                            self.ebar.append(localE)
                    except:
                        print(' not enough found samples of shape =%s / within label  =%s' % (shape, i_))
                        try:
                            sample_idx = np.random.choice(shape_idx_by_label.numpy(), size=min(r, count_by_label),
                                                          replace=True)
                            print('sample idx:', sample_idx)
                            for idx in sample_idx:
                                C = graphs[idx_by_labels[i_][idx]].clone().to(self.device)
                                noise_distrib_C = torch.distributions.normal.Normal(loc=0., scale=C.std() + 1e-05)
                                noise_C = noise_distrib_C.rsample(C.size()).to(self.device)
                                if torch.all(C == C.T):
                                    noise_C = (noise_C + noise_C) / 2
                                F = features[idx_by_labels[i_][idx]].clone().to(self.device)
                                noise_distrib_F = torch.distributions.normal.Normal(loc=torch.zeros(F.shape[-1]),
                                                                                 scale=F.std(axis=0) + 1e-05)
                                noise_F = noise_distrib_F.rsample(torch.Size([shape])).to(self.device)
                                new_C = C + noise_C
                                new_C.to(self.dtype)
                                new_C.requires_grad_(True)
                                new_F = F + noise_F
                                new_F.to(self.dtype)
                                # new_F.requires_grad_(True)
                                self.Cbar.append(new_C)
                                self.Fbar.append(new_F)
                        except:
                            print(' NO sample found with proper shape within label = %s' % i_)
                            # We generate a random graph from the distribution of graphs within this label
                            # Randomly over C
                            # by kmeans over F
                            local_idx = idx_by_labels[i_]
                            means_C = torch.tensor([graphs[idx].mean() for idx in local_idx])
                            distrib_C = torch.distributions.normal.Normal(loc=means_C.mean(), scale=means_C.std() + 1e-05)
                            for _ in range(min(r, count_by_label)):
                                C = distrib_C.rsample((shape, shape)).to(self.device)
                                C = (C + C.T) / 2.
                                if atoms_projection == 'clipped':
                                    new_C = (C - C.min()) / (C.max() - C.min())
                                elif atoms_projection == 'nonnegative':
                                    new_C = C.clamp(0)
                                new_C.to(self.dtype)
                                new_C.requires_grad_(True)
                                self.Cbar.append(new_C)
                                new_F = F_clusters.clone().to(self.device)
                                new_F.to(self.dtype)
                                self.Fbar.append(new_F)

                        continue

                # Embed template features through GIN layers
                # stack features for the sampled initial templates at each GIN layers :
                with torch.no_grad():
                    for idx_atom in range(self.Katoms):
                        F = self.Fbar[idx_atom].to(self.device)
                        edge_attr = self.ebar[idx_atom].to(self.device)
                        processedC = self.Cbar[idx_atom].to(self.device)
                        edge_index = torch.argwhere(processedC == 1.).T
                        embedded_features = self.gnn.encoder.get_layered_feature(F, edge_index, edge_attr)
                        embedded_features.requires_grad_(True)
                        self.Fbar[idx_atom] = embedded_features
                    print('processed Fbar shapes:', [F.shape for F in self.Fbar])
        self.atoms_params = [*self.Fbar]
        if self.learn_hbar:
            self.atoms_params += [*self.hbar]

        self.params = self.atoms_params + list(self.gnn.parameters()) + list(self.classifier.parameters())

        self.shape_atoms = [C.shape[0] for C in self.Cbar]


    def set_model_to_train(self):
        self.gnn.train()
        self.classifier.train()

    def set_model_to_eval(self):
        self.gnn.eval()
        self.classifier.eval()

    def compute_pairwise_euclidean_distance(self, list_F, list_Fbar, detach=True):
        list_Mik = []
        dim_features = list_F[0].shape[-1]
        for F in list_F:
            list_Mi = []
            F2 = F ** 2
            ones_source = torch.ones((F.shape[0], dim_features), dtype=self.dtype, device=self.device)
            for Fbar in list_Fbar:
                shape_atom = Fbar.shape[0]
                ones_target = torch.ones((dim_features, shape_atom), dtype=self.dtype, device=self.device)
                first_term = F2 @ ones_target
                second_term = ones_source @ (Fbar ** 2).T
                Mi = first_term + second_term - 2 * F @ Fbar.T
                if detach:
                    list_Mi.append(Mi.detach().cpu())
                else:
                    list_Mi.append(Mi)

            list_Mik.append(list_Mi)
        return list_Mik

    def compute_w_distances(self, batch_graphs, batch_embedded_features_uncat, batch_masses):
        w_distances = torch.zeros((len(batch_graphs), self.Katoms), dtype=self.dtype, device=self.device)
        list_Mij = self.compute_pairwise_euclidean_distance(batch_embedded_features_uncat, self.Fbar, detach=False)
        for i in range(len(batch_graphs)):
            for j in range(self.Katoms):
                if batch_masses[i].shape[0]!=list_Mij[i][j].shape[0]:
                    print(batch_masses[i].shape)
                    print(list_Mij[i][j].shape)
                w_distances[i,j] = ot.emd2(batch_masses[i], self.hbar[j], list_Mij[i][j])

        return w_distances

    def compute_w_distances_pruned(self, batch_graphs, batch_embedded_features_uncat, batch_masses, pruned_Fbar, pruned_hbar):
        w_distances = torch.zeros((len(batch_graphs), self.Katoms), dtype=self.dtype, device=self.device)
        list_Mij = self.compute_pairwise_euclidean_distance(batch_embedded_features_uncat, pruned_Fbar, detach=False)
        for i in range(len(batch_graphs)):
            for j in range(self.Katoms):
                w_distances[i,j] = ot.emd2(batch_masses[i], pruned_hbar[j], list_Mij[i][j])

        return w_distances

    def forward(self, evaluate, *args, **kwargs):
        data = kwargs.get('data')
        batch_C, batch_F, batch_h = self.batch_to_list(data)
        if not evaluate:
            out_readout, batch_embedded_features_uncat = self.gnn(*args, **kwargs)
            dist_features = self.compute_w_distances(batch_C, batch_embedded_features_uncat, batch_h)
            dist_norm = torch.norm(dist_features, p=2, dim=1, keepdim=True)
            dist_features = dist_features / dist_norm

            pred = self.classifier(dist_features)
            return pred, dist_features
        else:
            self.set_model_to_eval()
            with torch.no_grad():
                if self.learn_hbar:
                    pruned_Fbar, pruned_hbar = self.prune_atoms()
                else:
                    pruned_Fbar, pruned_hbar = self.Fbar, self.hbar
                out_readout, batch_embedded_features_uncat = self.gnn(*args, **kwargs)
                dist_features = self.compute_w_distances_pruned(batch_C, batch_embedded_features_uncat, batch_h, pruned_Fbar, pruned_hbar)
                dist_norm = torch.norm(dist_features, p=2, dim=1, keepdim=True)
                dist_features = dist_features / dist_norm

                pred = self.classifier(dist_features)

            self.set_model_to_train()
            return pred


    def batch_to_list(self, data: Batch):
        with torch.no_grad():
            batch_C = []
            batch_Y = []
            batch_F = []
            ptr = data.ptr
            coo = to_scipy_sparse_matrix(data.edge_index)
            indices = np.vstack((coo.row, coo.col))
            values = coo.data
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape
            C = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
            for j in range(data.y.shape[0]):
                batch_C.append(C[ptr[j]:ptr[j + 1], ptr[j]:ptr[j + 1]].to(self.dtype).to(self.device))
                batch_Y.append(data.y[j])
                batch_F.append(data.x[ptr[j]:ptr[j + 1], :].to(self.dtype).to(self.device))

            batch_h = [torch.ones(F.shape[0], dtype=torch.float64, device=self.device) / F.shape[0] for F in batch_F]

        return batch_C, batch_F, batch_h


    def prune_atoms(self):
        pruned_hbar = []
        pruned_Fbar = []
        for i, h in enumerate(self.hbar):
            nonzero_idx = torch.argwhere(h>0)[:, 0]
            pruned_hbar.append(h[nonzero_idx])
            pruned_Fbar.append(self.Fbar[i][nonzero_idx,:])
        return pruned_Fbar, pruned_hbar

    def prune_parameters(self):
        with torch.no_grad():
            if self.learn_hbar:
                for h in self.hbar:
                    h[:] = probability_simplex_projection(h)


