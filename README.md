# Stochastic Weighted Function Norm Regularization

### Introduction
This code implements a regularization method for neural networks.
It provides a proof of concept on a simple task: train a classifier on a small subset of MNIST

It is based on a work where we attempt to bridge the gap between statistical learning theory and deep neural network
developments. While the former provides solid foundations for regularization of several learning algorithms such as SVMs
and logistic regression, deep neural networks still suffer from a lack of systematic and mathematically motivated
regularizers. In this context, we propose to use a regularizer inspired from classical learning theory: Function norm.
In other words, we limit the hypothesis set in which we optimize the objective to a ball in the L2 function space.
As we proved that the exact computation of this norm is NP-hard, we propose to estimate it stochastically by generating
samples using a variational autoencoder.

### Prerequisites
The code has been tested under Fedora 27 using
1. Python 2.7.11
2. PyTorch 0.2.0_4

## Experiments
To run the experiments, refer to train_MNIST.py
We test with two architecture, one with which it is customary to use Dropout and one with which it is customary to use
Batch Normalization (see notebooks for details about the architectures and training parameters)

The results of our experiments are displayed in the notebooks. It appears that:
1. Dropout alone does better that our method, but using the combination gives the best results
2. Our method slightly outperforms batch normalization alone but the combination of both methods degrades the performance
3. In all cases our method alone outperforms substantially weight decay alone

## Contents
We provide 2 modules:

### VAE:
Defines 2 variational autorencoders architectures. For our experiments, we used only one type of VAE; Vanilla VAE.
The other architecture is given for further experimentation.

### MNIST:
Provides needed classes for random subset selection and models definition

We also provide some utils needed in our training code (see utils.py and pytorch_utils.py).

## Authors

* Amal Rannen Triki, Maxim Berman & Matthew B. Blaschko

** All the authours are with KU Leuven, ESAT-PSI, IMEC, Belgium
** A. Rannen Triki is also affiliated with Yonsei University, Korea

** For questions about the code, please contact Amal Rannen Triki (amal.rannen@esat.kuleuven.be)

## Citation


## Citation
```bibtex
@inproceedings{rannen2019function,
  title={Function norms for neural networks},
  author={Rannen-Triki, Amal and Berman, Maxim and Kolmogorov, Vladimir and Blaschko, Matthew B},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}

```


