from omegaconf import OmegaConf
from transformers import Conv1D
from layers import kn_pr_Conv1D
from layers import kn_Embedding
from layers import TQCompressedGPT2Config, KronAttention

import torch.nn as nn
import transformers


def get_model_cfg(cfg_path, model, compute=True, q_num=-1):
    cfg = OmegaConf.load(cfg_path)
    for k in cfg.layers.keys():
        if cfg.layers[k]["q_num"] != q_num and q_num != -1:
            continue
        name = k
        m = model
        names = name.split(".")
        for i in range(len(names) - 1):
            m = getattr(m, names[i])
        l = getattr(m, names[-1])
        if isinstance(l, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
            if compute:
                n_iter = cfg.layers[k]["n_iter"]
            else:
                n_iter = 0
            r = cfg.layers[k]["r"]
            m1 = cfg.layers[k]["m1"]
            n1 = cfg.layers[k]["n1"]
            m2 = cfg.layers[k]["m2"]
            n2 = cfg.layers[k]["n2"]
            config = TQCompressedGPT2Config(
                emb_compression_rate=1,
                attn_dec_shapes=(r, (m1, n1), (m2, n2)),
                mlp_dec_shapes=None,
                use_p=True,
                use_q=True,
            )

            layer_n = KronAttention.from_gpt2_attn(
                config=config,
                layer_idx=l.layer_idx,
                attn_weight_concat=l.c_attn.weight,
                attn_bias_concat=l.c_attn.bias,
                o_proj=l.c_proj,
                n_iter=n_iter,
            )
            setattr(m, names[-1], layer_n)
        elif isinstance(l, nn.Linear) or isinstance(l, Conv1D):
            if compute:
                n_iter = cfg.layers[k]["n_iter"]
            else:
                n_iter = 0
            use_p = cfg.layers[k]["use_p"]
            use_q = cfg.layers[k]["use_q"]
            setattr(
                m,
                names[-1],
                kn_pr_Conv1D.get_from_layer(
                    l,
                    cfg.layers[k]["r"],
                    cfg.layers[k]["m1"],
                    cfg.layers[k]["n1"],
                    cfg.layers[k]["m2"],
                    cfg.layers[k]["n2"],
                    use_p=use_p,
                    use_q=use_q,
                ),
            )
        elif isinstance(l, nn.Embedding):
            if compute:
                n_iter = cfg.layers[k]["n_iter"]
            else:
                n_iter = 0
            use_p = cfg.layers[k]["use_p"]
            setattr(
                m,
                names[-1],
                kn_Embedding.get_from_layer(
                    l,
                    d=cfg.layers[k]["d"],
                    use_p=use_p,
                ),
            )
        else:
            raise ValueError("Layer must be nn.Linear or Conv1D of embedding")
    return model
