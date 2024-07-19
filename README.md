# OCC-CLIP ECCV2024
This repository contains the code and data for the paper ["Model-agnostic Origin Attribution of Generated Images with Few-shot Examples"](https://arxiv.org/pdf/2404.02697) accepted by ECCV 2024.
## Overview
Overview of OCC-CLIP
![OCC-CLIP Overview](Flowchat.png)

# 🗓 Coming Soon
- [ ] Code release of our [paper](https://arxiv.org/pdf/2404.02697)
- [ ] Complete Instructions

## Demo
A Simple Demo
![Simple Demo](Teaser.png)

## Requirements
The environment.yml file contains the necessary packages to run the code. You can create the environment using the following command:
```
conda env create -f environment.yml
```

## Train
To obtain the perturbation, you can use the following command:
```
python run.py --train --train_real coco_test --train_target sdv1 --train_set 1 --n_epoch 200 --lr 0.0001 --Use_Attack --epsilon 0.1 --n_shots 50 --save_path {PATH}
```

## Test
To run the inference, you can use the following command:
```
python run.py --train_real coco_test --train_target ${TRAINF} --test --test_target ${TRAINF} --test_other ${OTHERDATASET} --train_set ${TRAIN_SET} --n_epoch ${EPOCH} --lr ${LR} --Use_Attack --epsilon ${EPSILON} --n_shots ${NSHOTS} --save_path {PATH}
```


## Clarifications
This task is very sensitive. We ran it on A40.
You need to set 
'''
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''
to get same experimental results each time.

- Dataset used in the paper

    The default non-target dataset used in the paper is a subset of MSCOCO validation set. There are a total of 202,520 fake images generated by five different generative models, namely, Stable Diffusion Model, Latent Diffusion Model, GLIDE, Vector Quantized Diffusion, and GALIP.
    We provide the dataset in the dataset/ directory. Generated images can be downloaded from here: https://drive.google.com/drive/folders/0B_6QlNu0UvSmflVTclRQeEZtVGpJVWFXcGlaSkNVMFh5V1ZWVWdRa1d2SDdEdzNwc0NwOG8?resourcekey=0-Gff5JTYKYiyObeUz7e3FMA&usp=sharing

## Acknowledgement
We would like to thank the authors of the following repositories for their code: https://github.com/KaiyangZhou/CoOp

<!-- ## Citation -->
<!-- If you find this repository useful, please consider citing our paper:
```
@inproceedings{
luo2024an,
title={An Image Is Worth 1000 Lies: Transferability of Adversarial Images across Prompts on Vision-Language Models},
author={Haochen Luo and Jindong Gu and Fengyuan Liu and Philip Torr},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=nc5GgFAvtk}
}
``` -->
