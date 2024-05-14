#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
import ripserplusplus as rpp_py

from line_profiler import profile


# from gph.python import ripser_parallel

def lp_loss(a, b, p=2):
    return (torch.sum(torch.abs(a - b) ** p))


def get_indicies(DX, rc, dim, card):
    dgm = rc['dgms'][dim]
    pairs = rc['pairs'][dim]

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for i in range(len(pairs)):
        s1, s2 = pairs[i]
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1, :][:, l1]), [len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2, :][:, l2]), [len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(dgm[i][1] - dgm[i][0])

    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())

    # Output indices
    indices = indices[:4 * card] + [0 for _ in range(0, max(0, 4 * card - len(indices)))]
    return list(np.array(indices, dtype=np.compat.long))


@profile
def Rips(DX, dim, card, n_threads, engine):
    # Parameters: DX (distance matrix),
    #             dim (homological dimension),
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)
    if dim < 1:
        dim = 1

    if engine == 'ripser':
        DX_ = DX.numpy()
        DX_ = (DX_ + DX_.T) / 2.0  # make it symmetrical
        DX_ -= np.diag(np.diag(DX_))
        rc = rpp_py.run("--format distance --dim " + str(dim), DX_)
    elif engine == 'giotto':
        rc = ripser_parallel(DX, maxdim=dim, metric="precomputed", collapse_edges=False, n_threads=n_threads)

    all_indicies = []  # for every dimension
    for d in range(1, dim + 1):
        all_indicies.append(get_indicies(DX, rc, d, card))
    return all_indicies


class RTD_differentiable(nn.Module):
    def __init__(self, dim=1, card=50, mode='minimum', n_threads=25, engine='giotto'):
        super().__init__()

        if dim < 1:
            raise ValueError(f"Dimension should be greater than 1. Provided dimension: {dim}")
        self.dim = dim
        self.mode = mode
        self.card = card
        self.n_threads = n_threads
        self.engine = engine

    @profile
    def forward(self, Dr1, Dr2, immovable=None):
        # inputs are distance matricies
        d, c = self.dim, self.card

        if Dr1.shape[0] != Dr2.shape[0]:
            raise ValueError(f"Point clouds must have same size. Size Dr1: {Dr1.shape} and size Dr2: {Dr2.shape}")

        if Dr1.device != Dr2.device:
            raise ValueError(
                f"Point clouds must be on the same devices. Device Dr1: {Dr1.device} and device Dr2: {Dr2.device}")

        device = Dr1.device
        # Compute distance matrices
        #         Dr1 = torch.cdist(r1, r1)
        #         Dr2 = torch.cdist(r2, r2)

        Dzz = torch.zeros((len(Dr1), len(Dr1)), device=device)
        if self.mode == 'minimum':
            Dr12 = torch.minimum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr12), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr1), 1)),
                                 0)  # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)),
                                 0)  # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX
        else:
            Dr12 = torch.maximum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr12.T), 1), torch.cat((Dr12, Dr2), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)),
                                 0)  # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr2.T), 1), torch.cat((Dr2, Dr2), 1)),
                                 0)  # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX

        # dr1 = Dr1.detach().cpu().numpy()
        # dr2 = Dr2.detach().cpu().numpy()
        # print(
        #     f'DEBUG before Rips: ',
        #     f'Dr1.shape={Dr1.shape} ',
        #     f'Dr2.shape={Dr2.shape}',
        #     f'DX.shape={DX.shape}',
        #     f'Dr1.min={dr1.min()} Dr1.mean={dr1.mean()} Dr1.max={dr1.max()} Dr1.std={dr1.std()}',
        #     f'Dr2.min={dr2.min()} Dr2.mean={dr2.mean()} Dr2.max={dr2.max()} Dr2.std={dr2.std()}',
        #     f'num of Dr1 < Dr2: {np.mean(dr1 < dr2)}',
        #     f'num of unique elems of dr1: {np.unique(dr1).shape}',
        #     f'num of unique elems of dr2: {np.unique(dr2).shape}',
        #     sep='\n',
        # )

        # Compute vertices associated to positive and negative simplices
        # Don't compute gradient for this operation
        all_ids = Rips(DX.detach().cpu(), self.dim, self.card, self.n_threads, self.engine)
        all_dgms = []
        for ids in all_ids:
            # Get persistence diagram by simply picking the corresponding entries in the distance matrix
            tmp_idx = np.reshape(ids, [2 * c, 2])
            if self.mode == 'minimum':
                dgm = torch.hstack([torch.reshape(DX[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c, 1]),
                                    torch.reshape(DX_2[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c, 1])])
            else:
                dgm = torch.hstack([torch.reshape(DX_2[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c, 1]),
                                    torch.reshape(DX[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c, 1])])
            all_dgms.append(dgm)
        return all_dgms


class RTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine='giotto', mode='minimum', is_sym=True, lp=1.0, **kwargs):
        super().__init__()

        self.is_sym = is_sym
        self.mode = mode
        self.p = lp
        self.rtd = RTD_differentiable(dim, card, mode, n_threads, engine)

    @profile
    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations

        # print(
        #     f'DEBUG RTDLoss: ',
        #     f'x_dist.shape={x_dist.shape} ',
        #     f'z_dist.shape={z_dist.shape}',
        #     x_dist, z_dist,
        #     sep='\n',
        # )

        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz):  # different dimensions
            loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=self.p)
            if self.is_sym:
                loss_zx += lp_loss(rtd_zx[d][:, 1], rtd_zx[d][:, 0], p=self.p)
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss


class MinMaxRTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine='giotto', is_sym=True, lp=1.0, **kwargs):
        super().__init__()

        self.is_sym = is_sym
        self.p = lp
        self.rtd_min = RTD_differentiable(dim, card, 'minimum', n_threads, engine)
        self.rtd_max = RTD_differentiable(dim, card, 'maximum', n_threads, engine)

    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd_min(x_dist, z_dist, immovable=1) + self.rtd_max(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd_min(z_dist, x_dist, immovable=2) + self.rtd_max(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz):  # different dimensions
            loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=self.p)
            if self.is_sym:
                loss_zx += lp_loss(rtd_zx[d][:, 1], rtd_zx[d][:, 0], p=self.p)
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss


from losses.persistent_homology import compute_distance_matrix


class MyRTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine='giotto', mode='minimum', is_sym=True, lp=1.0,
                 normalization_data=None, enable_since_epoch=0):
        super().__init__()
        self.rtd_loss = RTDLoss(dim, card, n_threads, engine, mode, is_sym, lp)
        self.norm_constant = nn.Parameter(data=torch.ones(1), requires_grad=True)
        self.normalization_data = normalization_data
        self.enable_since_epoch = enable_since_epoch

    def forward(self, input_tensors, embeddings, epoch, *args, **kwargs):
        if epoch < self.enable_since_epoch:
            return 0.0

        x_dist = compute_distance_matrix(input_tensors)
        assert len(input_tensors.shape) == 4
        batch_size, ch, h, w = input_tensors.shape
        x_dist = x_dist / np.sqrt(ch * h * w)  # normalization as in RTD paper

        if self.normalization_data is not None:
            norm_mode = self.normalization_data['mode']
            if norm_mode == 'constant':
                x_dist = x_dist * self.normalization_data['constant']
            else:
                raise RuntimeError(f'Unknown normalization mode: {norm_mode}')

        z_dist = compute_distance_matrix(embeddings)
        z_dist = z_dist / np.sqrt(embeddings.shape[1])  # RTD mistakenly normalize here by z_dist.shape[1]
        z_dist = self.norm_constant * z_dist

        return self.rtd_loss(x_dist, z_dist)[-1]

    def summary(self):
        data = dict()
        data['rtd_loss_norm_constant'] = self.norm_constant.item()
        return data