import numpy as np
from getData import readMNIST

try:
    train_images, train_labels = readMNIST("mnist-dataset/train-labels.idx1-ubyte",
                                         "mnist-dataset/train-images.idx3-ubyte")
    test_images, test_labels = readMNIST("mnist-dataset/t10k-labels.idx1-ubyte",
                                        "mnist-dataset/t10k-images.idx3-ubyte")
    print(f"Training images shape : {train_images.shape}")
    print(f"Test images shape : {test_images.shape}")

except FileNotFoundError:
    print("Error: Make sure the MNIST data files are in the same directory as your script.")
except ValueError as e:
    print(f"Error: {e}")