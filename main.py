import numpy as np
from getData import readMNIST
import matplotlib.pyplot as plt

try:
    train_images, train_labels = readMNIST("mnist-dataset/train-labels.idx1-ubyte",
                                         "mnist-dataset/train-images.idx3-ubyte")
    test_images, test_labels = readMNIST("mnist-dataset/t10k-labels.idx1-ubyte",
                                        "mnist-dataset/t10k-images.idx3-ubyte")
    #train_images is a collection of 60000 28*28 arrays, while the test_images is a collection of 10000 28*28 arrays
    print(f"{train_images.shape, test_images.shape}")
    #Preprocessing

    #a. Flatten Images
    #We flatten images matrix which was a 3D matrix to a 2D Matrix with each row representing a sample and each column corresponding to
    #a feature of the data
    x_train = train_images.reshape(train_images.shape[0], -1) #The -1 tells NumPy to automatically calculate the second dimension (784).
    x_test = test_images.reshape(test_images.shape[0], -1)

    # b. Normalize pixels
    # Convert to float and scale to be between 0 and 1
    #We normalize the pixels in order to help the gradient descent algorithm to converge faster.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"{x_train.shape, x_test.shape}")




except FileNotFoundError:
    print("Error: Make sure the MNIST data files are in the same directory as your script.")
except ValueError as e:
    print(f"Error: {e}")