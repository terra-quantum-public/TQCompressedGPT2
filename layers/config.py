from typing import Optional, Tuple
from transformers.models.gpt2 import GPT2Config

DecompositionShapes = Tuple[int, Tuple[int, int], Tuple[int, int]]


def check_decomposition_shape_match(dec_shapes: DecompositionShapes, nx: int, nf: int):
    """
    match decomposition shapes againts matrix
    :param dec_shapes: kronecker decomposition shapes
    :param nx: input feature dim
    :param nf: out feauture dim
    """
    _, (m1, n1), (m2, n2) = dec_shapes
    return m1 * m2 == nx and n2 * n1 == nf


class TQCompressedGPT2Config(GPT2Config):
    def __init__(
        self,
        emb_compression_rate: int = 2,
        attn_dec_shapes: Optional[DecompositionShapes] = None,
        mlp_dec_shapes: Optional[DecompositionShapes] = None,
        use_p: bool = True,
        use_q: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.emb_compression_rate = emb_compression_rate or 2

        attn_dec_shapes = attn_dec_shapes or (1, (384, 768), (2, 1))
        assert check_decomposition_shape_match(
            attn_dec_shapes, self.hidden_size, self.hidden_size
        )
        self.attn_decomposition_shapes = attn_dec_shapes

        mlp_dec_shapes = mlp_dec_shapes or (4, (384, 768), (2, 4))
        if self.n_inner is not None:
            inner_dim = self.n_inner
        else:
            inner_dim = 4 * self.hidden_size
        assert check_decomposition_shape_match(
            mlp_dec_shapes, self.hidden_size, inner_dim
        )
        self.mlp_dec_shapes = mlp_dec_shapes

        r, (m1, n1), (m2, n2) = mlp_dec_shapes
        self.mlp_dec_shapes_out = (r, (n1, m1), (n2, m2))

        self.use_p = use_p
        self.use_q = use_q
