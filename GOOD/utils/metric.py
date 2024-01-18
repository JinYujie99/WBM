r"""A metric function module that is consist of a Metric class which incorporate many score and loss functions.
"""

from math import sqrt
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score as sk_roc_auc, mean_squared_error, \
    accuracy_score, average_precision_score, mean_absolute_error, f1_score, confusion_matrix
from torch.nn.functional import cross_entropy, l1_loss, binary_cross_entropy_with_logits





class Metric(object):
    r"""
    Metric function module that is consist of a Metric class which incorporate many score and loss functions
    """


    def __init__(self):
        self.task2loss = {
            'Binary classification': binary_cross_entropy_with_logits,
            'Multi-label classification': self.cross_entropy_with_logit,
            'Regression': l1_loss
        }
        self.score_name2score = {
            'RMSE': self.rmse,
            'MAE': mean_absolute_error,
            'Average Precision': self.ap,
            'F1': self.f1,
            'ROC-AUC': self.roc_auc_score,
            'Accuracy': self.acc,
            'MCC': self.mcc
        }
        self.loss_func = self.cross_entropy_with_logit
        self.score_func = self.roc_auc_score
        self.dataset_task = ''
        self.score_name = ''

        self.lower_better = -1

        self.best_stat = {'score': None, 'loss': float('inf')}
        self.id_best_stat = {'score': None, 'loss': float('inf')}

    def set_loss_func(self, task_name, config):
        r"""
        Set the loss function

        Args:
            task_name (str): name of task

        Returns:
            None

        """
        self.dataset_task = task_name
        if task_name == 'Binary classification':
            self.loss_func = binary_cross_entropy_with_logits
        elif task_name == 'Multi-label classification':
            self.loss_func = self.cross_entropy_with_logit
        elif task_name == 'Regression':
            self.loss_func = l1_loss
        else:
            self.loss_func = weighted_cross_entropy(config.dataset.dataset_name)

        assert self.loss_func is not None

    def set_score_func(self, metric_name):
        r"""
        Set the metric function

        Args:
            metric_name: name of metric

        Returns:
            None

        """
        self.score_func = self.score_name2score.get(metric_name)
        assert self.score_func is not None
        self.score_name = metric_name.upper()
        if self.score_name in ['RMSE', 'MAE']:
            self.lower_better = 1
        else:
            self.lower_better = -1

    def f1(self, y_true, y_pred):
        r"""
        Calculate F1 score

        Args:
            y_true (torch.tensor): input labels
            y_pred (torch.tensor): label predictions

        Returns (float):
            F1 score

        """
        true = torch.tensor(y_true)
        pred_label = torch.tensor(y_pred)
        pred_label = pred_label.round() if self.dataset_task == "Binary classification" else torch.argmax(pred_label,
                                                                                                            dim=1)
        return f1_score(true, pred_label, average='micro')

    def ap(self, y_true, y_pred):
        r"""
        Calculate AP score

        Args:
            y_true (torch.tensor): input labels
            y_pred (torch.tensor): label predictions

        Returns (float):
            AP score

        """
        return average_precision_score(torch.tensor(y_true).long(), torch.tensor(y_pred))

    def roc_auc_score(self, y_true, y_pred):
        r"""
        Calculate roc_auc score

        Args:
            y_true (torch.tensor): input labels
            y_pred (torch.tensor): label predictions

        Returns (float):
            roc_auc score

        """
        return sk_roc_auc(torch.tensor(y_true).long(), torch.tensor(y_pred), multi_class='ovo')

    def reg_absolute_error(self, y_true, y_pred):
        r"""
        Calculate absolute regression error

        Args:
            y_true (torch.tensor): input labels
            y_pred (torch.tensor): label predictions

        Returns (float):
            absolute regression error

        """
        return mean_absolute_error(torch.tensor(y_true), torch.tensor(y_pred))

    def acc(self, y_true, y_pred):
        r"""
        Calculate accuracy score

        Args:
            y_true (torch.tensor): input labels
            y_pred (torch.tensor): label predictions

        Returns (float):
            accuracy score

        """
        true = torch.tensor(y_true)
        pred_label = torch.tensor(y_pred)
        pred_label = pred_label.round() if self.dataset_task == "Binary classification" else torch.argmax(pred_label,
                                                                                                            dim=1)
        return accuracy_score(true, pred_label)

    def mcc(self, y_true, y_pred):
        true = torch.tensor(y_true)
        pred_label = torch.tensor(y_pred)
        pred_label = pred_label.round() if self.dataset_task == "Binary classification" else torch.argmax(pred_label,
                                                                                                            dim=1)
        C = confusion_matrix(true, pred_label, sample_weight=None)
        t_sum = C.sum(axis=1, dtype=np.float64)
        p_sum = C.sum(axis=0, dtype=np.float64)
        n_correct = np.trace(C, dtype=np.float64)
        n_samples = p_sum.sum()
        cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
        cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
        cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
        mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)

        if np.isnan(mcc):
            return 0.
        else:
            return mcc

    def rmse(self, y_true, y_pred):
        r"""
        Calculate RMSE

        Args:
            y_true (torch.tensor): input labels
            y_pred (torch.tensor): label predictions

        Returns (float):
            RMSE

        """
        return sqrt(mean_squared_error(y_true, y_pred))

    def cross_entropy_with_logit(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        r"""
        Calculate cross entropy loss

        Args:
            y_pred (torch.tensor): label predictions
            y_true (torch.tensor): input labels
            **kwargs: key word arguments for the use of :func:`~torch.nn.functional.cross_entropy`

        Returns:
            cross entropy loss

        """
        return cross_entropy(y_pred, y_true.long(), **kwargs)

class Loss(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class weighted_cross_entropy(Loss):
    def __init__(self, dataset_name):
        weight = {
            "NCI1": [1 / 0.6230, 1 / 0.3770],
            "NCI109": [1 / 0.6204, 1 / 0.3796],
            "PROTEINS": [1 / 0.4197, 1 / 0.5803],
            "DD": [1 / 0.3547, 1 / 0.6453]
        }
        self.weight = weight.get(dataset_name, None)

    def __call__(self, y_pred, y_true, reduction):
        weight = torch.tensor(self.weight).to(y_pred.device).to(torch.float64) if self.weight is not None else None
        return cross_entropy(y_pred, y_true.long(), weight=weight, reduction=reduction)

