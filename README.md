# MNIST Handwritten Digit Classification Project
This repository contains a DevSoc AI ML Core project 3 focused on the classification of handwritten digits from the MNIST dataset. 
 - The project leverages only the use of Pandas, Numpy, Matplotlib and Mathematics behind implementation of neural networks

## Data Source
The project utilizes the MNIST dataset, which is publicly available on Kaggle at the following URL: https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data
 - The test.csv and train.csv was already provided in the google drive folder

## Project Overview
The core of the project is to build a neural network from scratch and train a neural network to predict the hand written digits. 

## Objective
Build a model that can accurately classify handwritten digits by implementing all the fundamental components of a neural network manually. This includes:
 - Creating custom structures for neurons and layers
 - Writing code for forward propagation
 - Implementing backpropagation to compute gradients and update weights
 - Experimenting with different activation functions and loss functions
 - Training using gradient descent and observing how various hyperparameters affect performance

## Core Components Implemented
 - Weights matrices are randomly initialized and bias matrices are started with zero
 - Input layer, hidden layer and output layers are implemented for network architecture
 - Forward prop and back prop are implemented which leverages ReLU activation and Crossentropy Loss
 - The training part uses mini-batch SGD optimization

## Learning Outcomes
 - Understand the mechanics behind neural networks
 - Learn how forward and backward propagation work internally
 - Explore the impact of different loss functions, optimizers, and activations
 - Build a strong foundation for diving deeper into deep learning and AI

## Data Acquisition and Preparation: 
Ensure that the data files are placed in the same directory as the Jupyter Notebook for data loading and processing.
