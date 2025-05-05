# Neural Network From Scratch

A high-performance neural network implementation built entirely from scratch in C++ with parallel computing capabilities and advanced optimization techniques.

## Project Overview

This project implements a feed-forward neural network with customizable architecture (currently configured as 784-128-64-10), designed for image classification tasks such as MNIST. The implementation features thread pooling for parallel computation, custom tensor operations, and multiple optimization algorithms.

## Key Features

- **Custom Tensor Implementation**: PyTorch-inspired tensor operations with strides and storage
- **Parallel Processing**: Thread pool implementation for distributed mini-batch processing
- **Multiple Optimizers**: Both SGD and Adam optimization algorithms
- **Weight Initialization**: He initialization for improved training dynamics
- **Model Serialization**: Save and load trained model weights
- **Configurable Architecture**: Easily adjustable network topology

## Technical Implementation

### Tensor Operations

The tensor implementation mimics PyTorch's approach using strides and storage mechanisms, allowing for efficient memory management and mathematical operations. This provides a solid foundation for the computational graph required for neural network operations.

### Network Architecture

Current configuration: 784 → 128 → 64 → 10
- Input layer: 784 neurons (compatible with 28×28 MNIST images)
- Hidden layer 1: 128 neurons
- Hidden layer 2: 64 neurons
- Output layer: 10 neurons (for digit classification)

The network uses cross-entropy loss with softmax activation for classification tasks.

### Optimization

Two optimization strategies are implemented:

1. **Stochastic Gradient Descent (SGD)**
   - Basic but effective optimization algorithm
   - Configurable learning rate and mini-batch size

2. **Adam Optimizer**
   - Adaptive learning rates for different parameters
   - Incorporates first and second moments of gradients
   - Provides faster convergence and better performance

### Parallelization

The implementation leverages thread pooling to distribute computation across available CPU cores:
- Each thread processes a portion of the mini-batch
- Thread synchronization ensures proper gradient aggregation
- Significant performance improvements on multi-core systems

## Performance Analysis

Test was done on [mnist data set](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

### SGD Performance
- **Mini-batch size**: 48
- **Single core**: 336 seconds per epoch, 81.76% efficiency
- **10 threads on 10-core system**: ~60 seconds per epoch (5.6× speedup)

### Adam Performance
- **10 threads on 10-core system**: Similar execution time to SGD
- **Efficiency**: 91.13% after just one epoch (higher than SGD)

## Implementation Insights

- The thread pool implementation efficiently distributes the computational load across all available cores, achieving near-linear speedup.
- He initialization helps address vanishing/exploding gradients, enabling training of deeper networks.
- The custom tensor implementation with strides provides memory efficiency while maintaining computational flexibility.
- Adam optimization provides superior convergence characteristics compared to standard SGD.
