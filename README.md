# MEFAttack

## Overview
This repository contains the official PyTorch implementation of our paper **Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**. We propose a novel **Maximin Expected Flatness (MEF) attack** that crafts transferable adversarial examples by enhancing loss surface flatness, achieving superior performance against both standard and defended models. Key Results on ImageNet:

- 🚀 **Computational Efficiency**  
  At **50% computational cost** (backward passes), MEF consistently outperforms SOTA method [PGN](https://github.com/Trustworthy-AI-Group/PGN) by **+4% average success rate** across 22 heterogeneous models

- ⚡ **Performance Dominance**  
  Under equivalent computational budgets, MEF achieves **+8% absolute improvement** over [PGN](https://github.com/Trustworthy-AI-Group/PGN)

- 🔥 **Augmentation Synergy**  
  When integrated with input transformations, MEF attains **additional 15% success rate gains** against defense mechanisms, setting new state-of-the-art robustness benchmarks

- ✅ ​**​Independent Verification​**  
  ​Validated by [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) framework, MEF ​​surpasses all 22 gradient-based attacks​​ in the benchmark, establishing itself as the **new SOTA in adversarial transferability**

## Motivation  
While flatness-enhanced attacks improve adversarial transferability by optimizing loss landscape geometry, existing methods suffer from fragmented flatness definitions and unproven theoretical connections, leading to suboptimal performance with high computational costs. We address these limitations by establishing the first theoretical link between multi-order flatness and transferability, revealing zeroth-order flatness as the dominant transferability source. Our Maximin Expected Flatness (MEF) attack leverages this insight through gradient-guided conditional sampling and balanced optimization.

## Requirements
* joblib==1.3.2
* numpy==1.24.4
* pandas==2.0.3
* Pillow==8.4.0
* Pillow==11.2.1
* pretrainedmodels==0.7.4
* timm==0.9.11
* torch==1.12.0
* torchvision==0.13.0
* tqdm==4.66.5

## Quick Start
### Prepare the dataset and models.
1. You can download the ImageNet-compatible dataset from [here](https://github.com/yuyang-long/SSA/tree/master/dataset) and put the data in **'./dataset/'**.

2. The normally trained models (i.e., Inc-v3, Inc-v4, IncRes-v2, Res-50, Res-101) are from "pretrainedmodels". 

3. The adversarially trained models (i.e, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2) are from [SSA](https://github.com/yuyang-long/SSA) or [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model). For more detailed information on how to use them, visit these two repositories. You can download the torch_nets_weights and put it to **'./model/'**.

### Runing attack
1. You can run our proposed attack as follows. 
```
python main.py --mode="attack"
```
2. The generated adversarial examples would be stored in the directory **./results**. Then run the file **main.py** in evaluation mode to get the transfer attack success rate:
```
python main.py --mode="eval"
```