import torch
import torch.nn as nn


class kn_Embedding(nn.Module):
    """
    Args:
        vocab_size: int
        embedding shape: int
        d: int
    """

    def __init__(self, vocab_size, emb_shape, d, use_p=True):
        super().__init__()
        n = emb_shape // d
        if n * d != emb_shape:
            raise ValueError("d must be divided by embedding shape")
        self.embedding = nn.Embedding(vocab_size, n)
        self.B = nn.Parameter(torch.randn(1, d))
        self.p = self.register_buffer("p", torch.randperm(emb_shape))
        self.use_p = use_p

    @classmethod
    def get_from_layer(
        cls,
        layer,
        d=1,
        use_p=True,
    ):
        assert isinstance(layer, nn.Embedding)
        w = layer.weight
        w = w.detach()
        vocab_size = w.shape[0]
        emb = w.shape[1]

        m1, m2, n1, n2 = vocab_size, 1, int(emb // d), d
        assert vocab_size == m1 * m2
        assert emb == n1 * n2

        kn_p_layer = cls(vocab_size, emb, d, use_p=use_p)
        kn_p_layer.p = torch.randperm(emb)
        return kn_p_layer

    def forward(self, x):
        x = self.embedding(x)
        if self.use_p:
            x = torch.kron(x, self.B)[:, :, self.p]
        else:
            x = torch.kron(x, self.B)
        return x
