# Flow Gaussian Mixture Model (FlowGMM)
This repository contains a PyTorch implementation of the Flow Gaussian Mixture Model (FlowGMM) model from our paper

[Semi-Supervised Learning with Normalizing Flows ](https://invertibleworkshop.github.io/accepted_papers/pdfs/INNF_2019_paper_28.pdf)

by Pavel Izmailov, Polina Kirichenko, Marc Finzi and Andrew Gordon Wilson.

# Introduction

# Dependencies
* [PyTorch](http://pytorch.org/) version 1.0.1
* [torchvision](https://github.com/pytorch/vision/) version 0.2.1
* [tensorboardX](https://github.com/lanpa/tensorboardX)

# Usage

## Examples

```bash
python3 experiments/train_flows/train_semisup_cons.py --dataset=MNIST --data_path=data/mnist/ --label_path=data/labels/mnist/100_balanced_labels/10.npz --logdir=<LOGDIR> --ckptdir=<CKPTDIR> --save_freq=5000 --means=random --means_r=1. --num_epochs=3000
0 --label_weight=3 --consistency_weight=1. --consistency_rampup=1000 --lr=1e-5 --eval_freq=100 --flow=RealNVP
```

# References

* RealNVP: [github.com/chrischute/real-nvp](https://github.com/chrischute/real-nvp)
