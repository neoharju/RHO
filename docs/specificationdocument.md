# Specification Document (Määrittelydokumentti)

## RHO

RHO will be designed to absorb gray-scale images that will be used to train a
neural network for classification with MNIST in mind. 
The neural network will be programmed from
scratch without relying on external machine learning libraries. 
The project will be programmed in C++ and delivered as a terminal application that

1. reads the MNIST gray-scale images as data from disk
2. trains the neural network using the data
3. classifies handwritten images 0-9 based on the trained neural network weights
  

## The core of the work

The project will need implementations from scratch for the neural network
training, these include:
- data loading and preprocessing (parser, normalization, shuffling)
- weight initialization (randomization)
- dense-layer forward and backward passes
  - an activation function (ReLU)
  - a loss function (cross-entropy)
- a backpropagation algorithm tying the above together
- parameter update rule (SGD/Adam)
- gradient checking
- evaluation with statistics printing

The trained neural network will be able to classify MNIST images to an
optimistic over 90% accuracy; these will be measured against the test set.

## Problem
The problem is essentially multi-class classification of handwritten number images. Each input is a 28x28 pixel grayscale images containing one handwritten digit (0-9). The task is to assign each image to the correct digit class. 

## Inputs
The grayscale images are preprocessed, normalized and converted into a (N, 28x28) contiguous array.
  - Pixel values are normalized (x_norm = (x - x_mean) / std using all training
    set pixel x values only
  - The elements are shuffled
  - The contiguous array is fed to the neural network.

## O-analysis
### Time

| Operation | Complexity | Note |
|---|---|---|
| Loading the data | O(N·d) | d pixels per image |
| Forward pass, one sample | O(W) | one GEMM with vector addition: Z = XW + b, which is O(P^3) for
pixels |
| Backward pass, one sample | O(2·W) | two GEMMs |
| Parameter update | O(W) | 
| One mini-batch step | O(B·W) | where B is the size of the mini-batch |
| One epoch | O(N·W) | plus O(N) for the shuffling of elements |
| Full training | O(E·N·W) | where E is the number of Epochs |
| Evaluation on the test set | Θ(N_test·W) | |

Per-training step we can assume the backward pass is the slowest part. Also since if layer input `n_in` and `n_out` are doubled the weight matrix `W = n_in · n_out` grows quadratically.

### Space
| Item | Complexity | Notes |
|---|---|---|
| Training data  | O(N·d) | ~N·d·(datatype size in bytes) |
| Parameters | O(W) | ~W·(`float32`) |
| Gradients | P(W) |  ~W·(`float32`)|

Most of the space is simply taken by the data.

## Programming languages

The work is implemented in C++, built with CMake and Ninja, and compiled with GCC (Clang does not understand modules (yet)). Only the standard library is used in the numerical core; parallelization will use OpenMP.

For peer review, the author can comfortably evaluate projects written in `Python`, `C++`, and CUDA C++. 

## Affiliation
The author is affiliated with the physics department

## Expected sources to be used
[0] (Y. LeCun, C. Cortes, C. Burges. The MNIST Database of Handwritten Digits.)[http://yann.lecun.com/exdb/mnist/]
[1] (M. Nielsen. Neural Networks and Deep Learning.)[http://neuralnetworksanddeeplearning.com/]
[2] (Wikipedia: Backpropagation.)[https://en.wikipedia.org/wiki/Backpropagation]
[3] (Intel. Intrinsics Guide)[https://www.intel.com/content/www/us/en/docs/intrinsics-guide/]
[4] (S. Bjorkqvist. Wiriting automates tests for neural networks.)[https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/]
