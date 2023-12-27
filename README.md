# Introduction
TQCompressedGPT-2 is an advanced neural network model, offering a novel method for model compression through improved tensor decompositions. It addresses the challenges of computational and storage demands in NLP tasks, introducing a permutation-based enhancement to Kronecker decomposition, significantly reducing model size while maintaining performance.\
TQCompressedGPT2 © 2024 by Terra Quantum AG is licensed under CC BY-NC-ND 4.0. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/ \
Any entity who wishes to use this library for commercial purposes should contact info@terraquantum.swiss for more information. \
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)\
<img src="https://cdn-uploads.huggingface.co/production/uploads/6476003bbed7adbb05f8441f/jEKdKKFoEzlAbbI4NnokH.png" width="500">
# Features
**Model Size Reduction:** Compresses the GPT-2small model from 124 million to 81 million parameters.\
**Permutation-Based Enhancement:** Introduces a new permutation algorithm for matrix factorization, minimizing performance degradation.\
**Efficient Training Strategy:** Employs multi-step knowledge distillation with a fraction (3.1%) of the OpenWebText dataset.\
**Performance:** Outperforms DistilGPT-2 in comparative evaluations.\
<img src="https://cdn-uploads.huggingface.co/production/uploads/6476003bbed7adbb05f8441f/x1krVBC2RTZNDR0dynbRp.png" width="500">

## Permutation-Based Enhancement
In our work we employ permutation-based algorithm, which allows to achieve better decomposition approximation for weight matrices:\
<img src="https://cdn-uploads.huggingface.co/production/uploads/6476003bbed7adbb05f8441f/bM6KwfKWYBJjeX_xGw83C.png" width="500">

# Methodology
For more details about the techniques of TQCompressedGPT-2, refer to our paper: **(ADD LINK)TQCompressor: Improving Tensor Decomposition in Neural Networks via Permutations**\
**TQCompressed Decomposition:** Focuses on optimal permutation of weight matrices followed by Kronecker decomposition.\
**Knowledge Distillation:** Uses an iterative compression method coupled with knowledge distillation, enhancing performance.\
**Application:** Demonstrated on the GPT-2 model, showing its versatility and applicability to various neural network architectures.

# Usage
**Install:** run `pip install -e .`\
**NOTE:** Model was tested on torch==2.0.1+cu117 (CUDA 11.7). For detailed guide on installation [visit PyTorch Website](https://pytorch.org/)\
The model and code are publicly available at:
- [GitHub Repository](https://github.com/terra-quantum-io/TQCompressedGPT2)
- [HuggingFace Repository](https://huggingface.co/tq-ag/TQCompressedGPT2)

# Citation
If you find TQCompressedGPT-2 useful in your research, please cite the following paper:
```
@article{tqcompressedgpt2,
  title={TQCompressor: Improving Tensor Decomposition in Neural Networks via Permutations},
  author={Abronin, V., Naumov, A., Mazur, D., Bystrov, D., Tsarova, K., Melnikov, Ar., Oseledets, I., Dolgov, S., Brasher, R., Perelshtein, M.},
  journal={arXiv preprint arXiv:[insert_arxiv_id]},
  year={2023}
}
```

# Acknowledgments
- [Terra Quantum AG](https://terraquantum.swiss/), Kornhausstrasse 25, 9000 St. Gallen, Switzerland
- Project contributors and researchers.
