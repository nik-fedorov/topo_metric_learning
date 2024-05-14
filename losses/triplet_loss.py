from typing import Optional, Dict

import torch
from torch import nn

from oml.functional.losses import get_reduced
from oml.interfaces.miners import labels2list
from oml.utils.misc_torch import elementwise_dist

from oml.miners.cross_batch import TripletMinerWithMemory
from oml.miners.inbatch_hard_cluster import HardClusterMiner
from oml.miners.inbatch_hard_tri import HardTripletsMiner
from oml.miners.inbatch_nhard_tri import NHardTripletsMiner
from oml.miners.inbatch_all_tri import AllTripletsMiner

from omegaconf import OmegaConf

from torch import Tensor


################################# MINERS #################################

class MyNHardTripletsMiner:
    def __init__(self, n_positive, n_negative):
        self.miner = NHardTripletsMiner(
            n_positive=OmegaConf.to_container(n_positive),
            n_negative=OmegaConf.to_container(n_negative)
        )

    def sample(self, *args, **kwargs):
        return self.miner.sample(*args, **kwargs)


################################# LOSSES #################################


def curve_relu(x, gamma):
    gamma = float(gamma)
    assert gamma >= 1.0

    res = torch.clone(x)
    mask = (res < 0)
    res[mask] = 0.0
    res[torch.logical_not(mask)] **= gamma
    return res


def relu_threshold(x, t):
    t = float(t)
    assert t > 0

    res = torch.clone(x)
    res[res < 0] = 0.0
    res[res > t] = t
    return res


class TripletLoss(nn.Module):
    """
    Class, which combines classical `TripletMarginLoss` and `SoftTripletLoss`.
    The idea of `SoftTripletLoss` is the following:
    instead of using the classical formula
    ``loss = relu(margin + positive_distance - negative_distance)``
    we use
    ``loss = log1p(exp(positive_distance - negative_distance))``.
    It may help to solve the often problem when `TripletMarginLoss` converges to it's
    margin value (also known as `dimension collapse`).
    """

    criterion_name = "triplet"  # for better logging

    def __init__(self, margin: Optional[float], reduction: str = "mean", need_logs: bool = False):
        """
        Args:
            margin: Margin value, set ``None`` to use `SoftTripletLoss`
            reduction: ``mean``, ``sum`` or ``none``
            need_logs: Set ``True`` if you want to store logs
        """
        assert reduction in ("mean", "sum", "none")
        # assert (margin is None) or (margin > 0)

        super().__init__()

        self.margin = margin
        self.reduction = reduction
        self.need_logs = need_logs
        self.last_logs: Dict[str, float] = {}

        self.log_dap, self.log_dan, self.log_dpn = [], [], []

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        """
        Args:
            anchor: Anchor features with the shape of ``(batch_size, feat)``
            positive: Positive features with the shape of ``(batch_size, feat)``
            negative: Negative features with the shape of ``(batch_size, feat)``
        Returns:
            Loss value
        """
        assert anchor.shape == positive.shape == negative.shape

        positive_dist = elementwise_dist(x1=anchor, x2=positive, p=2)
        negative_dist = elementwise_dist(x1=anchor, x2=negative, p=2)
        pos_neg_dist = elementwise_dist(x1=positive, x2=negative, p=2)

        self.log_dap.append(positive_dist)
        self.log_dan.append(negative_dist)
        self.log_dpn.append(pos_neg_dist)

        if self.margin is None:
            # here is the soft version of TripletLoss without margin
            loss = torch.log1p(torch.exp(positive_dist - negative_dist))
        else:
            loss = torch.relu(self.margin + positive_dist - negative_dist)
            # loss = curve_relu(self.margin + positive_dist - negative_dist, gamma=2.0)
            # loss = relu_threshold(self.margin + positive_dist - negative_dist, t=self.margin + 0.45)

        if self.need_logs:
            self.last_logs = {
                "active_tri": float((loss.clone().detach() > 0).float().mean()),
                "pos_dist": float(positive_dist.clone().detach().mean().item()),
                "neg_dist": float(negative_dist.clone().detach().mean().item()),
            }

        loss = get_reduced(loss, reduction=self.reduction)

        return loss

    def summary(self):
        res = {}
        dap, dan, dpn = map(torch.cat, [self.log_dap, self.log_dan, self.log_dpn])
        diff = dan - dap
        for data, name in zip([dap, dan, dpn, diff],
                              ['d(a,p)', 'd(a,n)', 'd(p,n)', 'd(a,n)-d(a,p)']):
            res[name + '/min'] = torch.min(data).item()
            res[name + '/mean'] = torch.mean(data).item()
            res[name + '/std'] = torch.std(data).item()
            res[name + '/max'] = torch.max(data).item()
        self.log_dap, self.log_dan, self.log_dpn = [], [], []
        return res


class TripletLossWithMiner(nn.Module):
    def __init__(self, triplet_loss, miner):
        super().__init__()
        self.tri_loss = triplet_loss
        self.miner = miner

    def forward(self, embeddings, labels, *args, **kwargs):
        labels_list = labels2list(labels)
        anchor, positive, negative = self.miner.sample(features=embeddings, labels=labels_list)
        loss = self.tri_loss(anchor=anchor, positive=positive, negative=negative)
        return loss

    def summary(self):
        return self.tri_loss.summary()
