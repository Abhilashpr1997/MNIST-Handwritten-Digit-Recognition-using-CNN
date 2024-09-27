# MNIST-Handwritten-Digit-Recognition-using-CNN
## Project Overview
This project demonstrates how to build and train a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The MNIST dataset consists of 70,000 grayscale images of digits from 0 to 9, each 28x28 pixels in size. By using TensorFlow and Keras, we develop a deep learning model to classify these digits with high accuracy.

The project covers the following tasks:

Loading and preprocessing the MNIST dataset.
Building a CNN model with convolutional and pooling layers.
Training and evaluating the model on the test data.
Visualizing the accuracy and loss during training.

## Table of Contents:
1.Project Overview
2.Dataset
3.Model Architecture
4.Training the Model
5.Evaluation
6.Visualizations
7.Results

## Dataset
The MNIST dataset is a well-known benchmark for image classification tasks in machine learning. It consists of:

Training set: 60,000 images
Test set: 10,000 images
Each image represents a handwritten digit (0-9) and has a size of 28x28 pixels. The pixel values are normalized between 0 and 1 before feeding them into the CNN model.

## Model Architecture
The CNN model used for this project consists of several layers designed to extract features and classify the images:

Input Layer: 28x28 grayscale images.
Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation.
MaxPooling Layer 1: 2x2 pool size.
Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation.
MaxPooling Layer 2: 2x2 pool size.
Flatten Layer: Flattens the 2D output to a 1D vector.
Dense Layer: 128 units, ReLU activation.
Output Layer: 10 units (one for each digit), Softmax activation.

## Model Details:
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy

## Training the Model
The training script (train.py) performs the following steps:

Load the MNIST dataset using keras.datasets.
Normalize the pixel values to be between 0 and 1.
Build the CNN model with convolutional, pooling, and dense layers.
Compile the model using the Adam optimizer and categorical crossentropy loss function.
Train the model using the training data, and validate using validation data.
Plot the training/validation accuracy and loss over epochs.

## Evaluation
After training, the model is evaluated on the test dataset to measure its performance. The following metrics are used:

Test Accuracy: The percentage of correctly classified digits on the test set.
Test Loss: The categorical crossentropy loss on the test set.

## Visualizations
During training, both accuracy and loss for the training and validation datasets are tracked and plotted.

## Results
The model achieved the following performance on the MNIST dataset:

Training Accuracy: ~99%
Validation Accuracy: ~98%
Test Accuracy: ~98%
The model generalizes well to unseen data and is capable of classifying handwritten digits with high accuracy.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
