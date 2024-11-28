# Intro to AI Project (Digit Recognition)

## Overview
This project aims to create a digit recognition system using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and leverages TensorFlow and OpenCV for preprocessing and prediction. It achieves accurate classification of handwritten digits and supports saving the trained model for reuse.

### Step 1: Prepare Data Pipeline
Load the MNIST dataset, which includes 60,000 training images and 10,000 test images of digits (0–9). The dataset is accessed via `tf.keras.datasets`, divided into training and testing subsets.

### Step 2: Preprocessing Data
Normalize pixel values to the range [0, 1] to accelerate training. Reshape the grayscale images into dimensions (28, 28, 1) suitable for CNN input. Test images undergo similar preprocessing.

### Step 3: Building the CNN Model
Define a CNN using TensorFlow’s Keras API:
- Three convolutional layers (32, 64, 64 filters) with ReLU activation.
- MaxPooling layers to reduce spatial dimensions.
- Flatten and Dense layers for feature extraction and classification.
- Softmax activation in the output layer for digit probabilities.

Compile the model with Adam optimizer and sparse categorical cross-entropy loss.

### Step 4: Evaluating Performance
Evaluate the model on the test set using accuracy and a validation set during training. Create confusion matrices to understand misclassifications.

### Step 5: Saving the Model
Save the trained model in `.h5` format using `model.save()`. The saved model supports loading for future predictions.

## Requirements
- Python 3.7+
- TensorFlow
- OpenCV
- NumPy

## Model Explanation
The CNN features a three-layer convolutional architecture designed for feature extraction:
1. Each Conv2D layer applies filters to learn local patterns from digit images.
2. MaxPooling layers reduce feature size, preventing overfitting and improving efficiency.
3. Dense layers refine learned patterns into digit classifications, with the final softmax layer outputting probabilities for each class.

Training on MNIST, the model learns to distinguish handwritten digits with minimal error.

## Usage
1. Use the model:
   ```bash
   python DigitRecognition.py
