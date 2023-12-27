import torch
import torch.nn as nn
import numpy as np
from opt_einsum import contract


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class kn_mlp_Conv1D(nn.Module):
    """
    Args:
        nf = n1 * n2 (`int`): The number of output features.
        nx = m1 * m2 (`int`): The number of input features.
    """

    def __init__(self, r, m1, n1, m2, n2, act):
        super().__init__()
        A1 = torch.empty((r, n1, m1))
        B1 = torch.empty((r, m2, n2))
        A2 = torch.empty((r, m1, n1))
        B2 = torch.empty((r, n2, m2))
        nn.init.normal_(A1, std=0.001)
        nn.init.normal_(B1, std=0.001)
        nn.init.normal_(A1, std=0.001)
        nn.init.normal_(B1, std=0.001)

        self.A1 = nn.Parameter(A1)
        self.B1 = nn.Parameter(B1)
        self.A2 = nn.Parameter(A2)
        self.B2 = nn.Parameter(B2)

        self.b1 = nn.Parameter(torch.zeros(n1 * n2))
        self.b2 = nn.Parameter(torch.zeros(m1 * m2))
        self.p = torch.randperm(m1 * m2)
        q = torch.randperm(m1 * m2)
        self.register_buffer("q", q)
        self.m1 = m1
        self.n1 = n1
        self.m2 = m2
        self.n2 = n2
        self.r = r
        p_inv = inverse_permutation(self.p)
        self.register_buffer("p_inv", p_inv)
        self.act = act

    @classmethod
    def get_from_layer(
        cls,
        layer,
        r,
        m1,
        n1,
        m2,
        n2,
        act,
        verbose=True,
        n_iter_als=-1,
        n_iter=5,
        method_type="annealing",
        count_samples=3,
        m=0.5,
        tol=1e-1,
    ):
        params_ = []
        names_ = []
        for name, param in layer.named_parameters():
            params_.append(param)
            names_.append(".".join((name.split(".")[:-1])))
        t_ = getattr(layer, names_[0])
        w = params_[0]
        b = params_[1]
        if isinstance(t_, nn.Linear):
            w = w.t()
        w = w.detach()
        nx = w.shape[0]
        nf = w.shape[1]
        assert nx == m1 * m2
        assert nf == n1 * n2
        kn_p_layer = cls(r, m1, n1, m2, n2, act)
        kn_p_layer.q = torch.randperm(m1 * m2)
        kn_p_layer.p_inv = torch.randperm(m1 * m2)
        return kn_p_layer

    def forward(self, x, residual=None):
        torch.cuda.synchronize()
        shape = x.size()
        x = x[:, :, self.p_inv].view((-1, self.m2))  # (k,m2)
        # B - (r,m2,n2)
        Bx = contract("km,rmn->rkn", x, self.B1)
        # Bx = x @ self.B #(r,k,n2)

        # Bx - (r,batch,m1,n2)
        Bx = Bx.view((self.r, np.prod(shape[:-1]), self.m1, self.n2))
        # A - (r,n1,m1)
        ABx = contract("rnm,rbmj->bnj", self.A1, Bx)  # (batch,n1,n2)
        ABx = torch.reshape(ABx, shape[:-1] + (self.n1 * self.n2,)) + self.b1

        x = self.act(ABx)
        shape = x.size()
        x = x.view((-1, self.n2))
        Bx = contract("km,rmn->rkn", x, self.B2)
        Bx = Bx.view((self.r, np.prod(shape[:-1]), self.n1, self.m2))
        ABx = contract("rnm,rbmj->bnj", self.A2, Bx)
        ABx = (torch.reshape(ABx, shape[:-1] + (self.m1 * self.m2,)) + self.b2)[
            :, :, self.q
        ]
        if residual is not None:
            ABx += residual
        return ABx
