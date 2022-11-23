# Renet-pytorch
Pytorch implementation for paper [Renet (Relational embedding network) ](https://arxiv.org/abs/2108.09666).

## Requirements
- Python==3.9.6
- torch==1.7.1
- torchvision==0.8.2
- GPU: NVIDIA GeForce RTX 3090 * 1 

## 

## Mini-Imagenet-S
According to the split in [ICLR(Meta-Learning with Fewer Tasks through Task Interpolation)](http://arxiv.org/abs/2106.02695), I split the raw mini-Imagenet dataset to reduce its training classes by specific sequence.

## Dermnet-S
According to the split in [ICLR(Meta-Learning with Fewer Tasks through Task Interpolation)](http://arxiv.org/abs/2106.02695), I split the raw Dermnet dataset to reduce its training classes by specific sequence.

## Remove validation
According to the training process in [ICLR(Meta-Learning with Fewer Tasks through Task Interpolation)](http://arxiv.org/abs/2106.02695), I remove the validation process during every epoch of training.
