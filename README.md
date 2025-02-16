# Handwritten Digit Recognition (MNIST) in C++

## Overview
This project uses a TensorFlow model and OpenCV to recognize handwritten digits from the MNIST dataset.

## Requirements
- OpenCV
- TensorFlow C++ API
- CMake

## How to Build and Run
1. Compile using CMake:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

2. Run the program:
   ```bash
   ./mnist_digit_recognition mnist_model.pb test_image.png
   ```

## Model
Use a pre-trained TensorFlow model (`mnist_model.pb`). You can train your own using Keras and export it.
