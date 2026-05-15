---
license: mit
tags:
- transformers
language:
- en
pipeline_tag: any-to-any
---

# FaceXFormer Model Card

<div align="center">

[**Project Page**](https://kartik-3004.github.io/facexformer/) **|** [**Paper (ArXiv)**](https://arxiv.org/abs/2403.12960v2) **|** [**Code**](https://github.com/Kartik-3004/facexformer)


</div>

## Introduction

FaceXFormer is an end-to-end unified model capable of handling a comprehensive range of facial analysis tasks such as face parsing, 
landmark detection, head pose estimation, attributes prediction, age/gender/race estimation, facial expression recognition and face visibility prediction.

<div  align="center">
<img src='assets/intro.png'>
</div>

## Model Details

FaceXFormer is a transformer-based encoder-decoder architecture where each task is treated as a learnable token, enabling the 
integration of multiple tasks within a single framework.

<div  align="center">
<img src='assets/main_archi.png'>
</div>

## Usage

The models can be downloaded directly from this repository or using python:
```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir="./")
```

## Citation
```bibtex
@article{narayan2024facexformer,
  title={FaceXFormer: A Unified Transformer for Facial Analysis},
  author={Narayan, Kartik and VS, Vibashan and Chellappa, Rama and Patel, Vishal M},
  journal={arXiv preprint arXiv:2403.12960},
  year={2024}
}
```

Please check our [GitHub repository](https://github.com/Kartik-3004/facexformer) for complete inference instructions.