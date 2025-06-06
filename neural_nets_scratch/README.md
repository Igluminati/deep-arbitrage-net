# Neural Networks from Scratch

This directory contains a collection of Jupyter notebooks that guide you through the process of building and understanding neural networks from scratch. Each notebook focuses on different aspects of neural networks, from basic concepts to more advanced techniques.

## End-to-End Process Overview

The process of making a prediction with a neural network involves several key steps:

1. **Data Preparation**: Load and preprocess the dataset. This includes normalizing the data, handling missing values, and splitting the dataset into training and testing sets.

2. **Model Initialization**: Define the architecture of the neural network, including the number of layers, the type of layers (e.g., dense, convolutional), and the activation functions to use.

3. **Training**: Train the model using the training dataset. This involves:
   - **Forward Propagation**: Compute the output of the neural network by passing the input data through each layer.
   - **Loss Calculation**: Determine the error or loss between the predicted output and the actual values.
   - **Backward Propagation**: Calculate the gradients of the loss with respect to each weight in the network using the chain rule.
   - **Optimization**: Update the weights of the network using an optimization algorithm (e.g., SGD, Adam) to minimize the loss.

4. **Evaluation**: Assess the model's performance using the test dataset to ensure it generalizes well to unseen data.

5. **Prediction**: Use the trained model to make predictions on new, unseen data. This involves preprocessing the input data in the same way as the training data and passing it through the network to get the predicted output.

## Notebooks Overview

- **a_real_dataset.ipynb**: Introduces the Fashion MNIST dataset to demonstrate how to load and handle real-world data.

- **backpropagation.ipynb**: Explains the concept of backpropagation, taking gradients of loss with respect to the output layer and propagating them back through the network.

- **batch_training.ipynb**: Demonstrates how to split a dataset into small fixed-size batches and train the model on each batch sequentially.

- **batches_layers_and_objects.ipynb**: Initialises the setup and demonstrates the use of NumPy for handling data in batches.

- **better_optimisers_(rmsprop,_adam).ipynb**: Implements advanced optimisation techniques such as Root Mean Square Propagation (RMSprop) and Adaptive Moment Estimation (Adam).

- **binary_logistic_regression.ipynb**: Introduces Binary Logistic Regression, fitting a sigmoid curve for binary classification tasks.

- **coding_a_layer.ipynb**: Implements multiple neurons in a single layer, demonstrating how to structure a layer in a neural network.

- **full_code_implementation.ipynb**: Provides a comprehensive implementation of a neural network model, including layers, optimisers, loss functions, and utilities for training, evaluating, and predicting.

- **hidden_layer_activation_functions.ipynb**: Fixes package initialisation and explores different activation functions for hidden layers.

- **implementing_loss.ipynb**: Computes the Categorical Cross-Entropy (CCE) Loss on a batch of Softmax outputs, demonstrating how to implement loss functions.

- **intro_and_neuron_code.ipynb**: Computes the output of a single neuron with arbitrary arrays of inputs and weights, introducing basic neuron operations.

- **introducing_optimisation_and_derivatives.ipynb**: Covers the calculation of partial derivatives of neural network layer operations to understand optimisation.

- **l1_and_l2_regularisation_and_dropout.ipynb**: Implements L1 and L2 Regularisation, which are methods to penalise large weights and prevent overfitting, along with dropout techniques.

- **md_to_ipynb_converter.py**: A utility script to convert Markdown files to Jupyter Notebook format.

- **model_evaluation.ipynb**: Evaluates the model on training data at the end of the training process to return performance metrics.

- **model_object.ipynb**: Encapsulates and abstracts the neural network training logic into a reusable model object.

- **optimisers.ipynb**: Provides a theoretical overview and full implementation of optimisers like Stochastic Gradient Descent (SGD).

- **prediction_inference.ipynb**: Demonstrates how to make predictions using a trained neural network model, including preprocessing steps and prediction logic.

- **regression.ipynb**: Delves into regression tasks, determining a specific value based on input through regression techniques.

- **saving_and_loading_model_and_their_parameters.ipynb**: Covers saving and loading models and their parameters for reuse and further training.

- **softmax_activation.ipynb**: Implements the Softmax activation layer with exponentiation and normalisation for multi-class classification tasks.

- **the_dot_product.ipynb**: Uses the dot product to compute the outputs of the first layer, demonstrating the fundamental operation in neural networks.

## Usage

To explore these notebooks, you can open them in Jupyter Notebook or JupyterLab. Each notebook is designed to be self-contained, but they build upon concepts introduced in previous notebooks. It is recommended to follow them in the order listed to build a comprehensive understanding of neural networks from scratch.
