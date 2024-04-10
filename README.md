<p align="center">
<h2 align="center">  Living Scenes: Multi-object Relocalization <br> and Reconstruction in Changing 3D Environments </h2>

<p align="center">
    <a href="http://zhuliyuan.net/"><strong>Liyuan Zhu</strong></a><sup>1</sup>, 
    <a href="https://shengyuh.github.io/"><strong>Shengyu Huang</strong></a><sup>2</sup>,
    <a href="https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en"><strong>Konrad Schindler</strong></a><sup>2</sup>,
    <a href="https://ir0.github.io/"><strong>Iro Armeni</strong></a><sup>1</sup>
  </p>

<p align="center"><strong>CVPR 2024</strong></a>
<p align="center"><strong><sup>1</sup>Stanford University, <sup>2</sup>ETH Zurich</strong></a>
  <h3 align="center"><a href="https://arxiv.org/abs/2312.09138">Arxiv</a> 
  | <a href="https://zhuliyuan.net/livingscenes">Website</a> | <a href="https://github.com/GradientSpaces/LivingScenes?tab=readme-ov-file#training-data">Data</a> </h3> 
  <div align="center"></div>

This repository represents the official implementation of this paper.

## Abstract
<image src="misc/teaser.png"/>
</p>

<details>
Research into dynamic 3D scene understanding has primarily focused on short-term change tracking from dense
observations, while little attention has been paid to longterm changes with sparse observations. We address this
gap with MORE2
, a novel approach for multi-object relocalization and reconstruction in evolving environments. We
view these environments as “living scenes” and consider
the problem of transforming scans taken at different points
in time into a 3D reconstruction of the object instances,
whose accuracy and completeness increase over time. At
the core of our method lies an SE(3)-equivariant representation in a single encoder-decoder network, trained on synthetic data. This representation enables us to seamlessly
tackle instance matching, registration, and reconstruction.
We also introduce a joint optimization algorithm that facilitates the accumulation of point clouds originating from the
same instance across multiple scans taken at different points
in time. We validate our method on synthetic and real-world
data and demonstrate state-of-the-art performance in both
end-to-end performance and individual subtasks.
</details>


## Installation
The code has been tested on Ubuntu 22.04, Intel 13900K, Nvidia 4090 24 GB/A100 80G. 

### Set up conda environment
```
conda create -n livingscenes python=3.9 -y
conda activate livingscenes
```

### Install key packages
Feel free to adapt the package versions based on your hardware setting.
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d=0.7.4 -c pytorch3d -y
conda install pyg -c pyg -y
```

### Install pip dependency
```
pip install cython
cd lib_shape_prior
python setup.py build_ext --inplace
pip install -U python-pycg[all] -f https://pycg.huangjh.tech/packages/index.html
pip install -r requirements.txt
```
If your hardware is the same as the tested ones, just run `install.sh`.

## Dataset Preparation

### Training Data
We generate the training data following [EFEM](https://github.com/JiahuiLei/EFEM). It will take three steps:

1. Make the ShapeNet mesh watertight
2. Generate SDF samples from the watertight mesh
3. Render depth maps and back-project them to point clouds

We provide the script to process the mesh: https://github.com/Zhu-Liyuan/mesh_processing.

### FlyingShape
The FlyingShape dataset is generated from [ShapeNet](https://shapenet.org/). [[download link](https://drive.google.com/file/d/1FPkXtNynYqhIDWOdYNEyEKTfYJMsknNa/view?usp=drive_link)].

### 3RScan
Please download the dataset from the original repository https://github.com/WaldJohannaU/3RScan. The authors provide tools to process the raw data.

We use the raw RGB-D measurements and back-project the depth images to get the point cloud of the scene. We also filter out the background.

### Change the path in config
When you finish downloading the dataset, change the `root_path` in `configs/3rscan.yaml` based on where you put the data on your machine.

## Train Shape Prior
To train the VN encoder-decoder network on ShapeNet data, run
```
cd lib_shape_prior
python run.py --config configs/3rscan/dgcnn_attn_inner.yaml
```

## Evaluation
We also provide the pretrained weight from our paper in `weight` folder. You should be able to reproduce similar performance in the paper.

To evalaute performance on 3RScan, run:
```
python eval_3rscan.py
```

To evaluate performance on FlyingShape, run:
```
python eval_flyingshape.py
```

## Using Predicted Segmentation (Mask3d)
Thanks to the authors of [Mask3D](https://jonasschult.github.io/Mask3D/), you can use their updated code to get predicted segmentation of your point cloud and run LivingScenes on it. [https://github.com/cvg/Mask3D]

## Contact
If you have any question, please contact Liyuan Zhu (liyuan.zhu@stanford.edu).

## Ackownledgement
Our implementation on the shape prior training part heavily relies on [EFEM](https://github.com/JiahuiLei/EFEM) and [Vector Neurons](https://github.com/FlyingGiraffe/vnn/) and we thank the authors for open sourcing their code and their insightful discussion at the early stage of this project. We also thank Francis Engelmann for providing the updated Mask3D. So please cite them as well.

## Citation
If you find our code and paper useful, please cite
```bibtex
@inproceedings{zhu2023living,
author = {Liyuan Zhu and Shengyu Huang and Konrad Schindler, Iro Armeni},
title = {Living Scenes: Multi-object Relocalization and Reconstruction in Changing 3D Environments},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2024}
}
```
