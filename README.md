> [new] A colab demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/) of ECO-TR is provided here.

### [Project Page](https://dltan7.github.io/ecotr/)

> ECO-TR: Efficient Correspondences Finding Via Coarse-to-Fine Refinement, ECCV2022
>
> Dongli Tan\*,  Jiang-Jiang Liu\*,  Xingyu Chen, Chao Chen, Ruixin Zhang, Yunhang Shen, Shouhong Ding, Rongrong Ji 

## Abstract

![pipeline](D:\code\ECO-TR\project_page\ecotr_page\ecotr\static\pics\pipeline.png)

We propose an efficient structure named Efficient Correspondence Transformer ECO-TR by finding correspondences in a coarse-to-fine manner, which significantly improves the efficiency of functional model methods. To achieve this, multiple transformer blocks are stage-wisely connected to gradually refine the predicted coordinates upon a shared multi-scale feature extraction network. All the correspondences are predicted within a single feed-forward pass, given a pair of images and for arbitrary query coordinates. We further propose an adaptive query-clustering strategy and an uncertainty-based outlier detection module to cooperate with the proposed framework for faster and better predictions. Experiments on various sparse and dense matching tasks demonstrate the superiority of our method in both efficiency and effectiveness against existing functional matching methods.

## Get started

Clone the repo:

> git clone https:////github.com/dltan7/ECO-TR.git

download model weights from [here](https://drive.google.com/file/d/1-1r7DQRHWvJDQfMkDs6m2a97YqzLCJvI/view?usp=sharing)

and prepare the environment below:

```shell
conda env create -f environment.yaml
conda activate eco-tr
```

## Running

### Match image pairs with ECO-TR

- Matching demo

An example is given in `demos/match_pair.py`.

- notebook

Two notebook examples are given in `notebooks`.



## Evaluation

This repo is under refactoring. We will release other codes later.

Part of the model and evaluation codes are borrowed or ported from [COTR](https://github.com/ubc-vision/COTR), [LoFTR](https://github.com/zju3dv/LoFTR), and [DenseMatching](https://github.com/PruneTruong/DenseMatching). Please also cite these works if you find the corresponding code useful.











