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

    #Visualizing the actual as well as the preprocessed images
    print("\nDisplaying a few sample images...")

    _, axes = plt.subplots(1, 10, figsize=(15, 2))  # Create a figure with 1 row, 10 columns of subplots
    for i in range(10):
        # Display the original (non-flattened, non-normalized) image
        # We take the first 10 images from the original train_images array
        # train_images[i] is a (28, 28) array
        axes[i].imshow(train_images[i], cmap='gray')
        axes[i].set_title(f"Label: {train_labels[i]}")  # Display its corresponding label
        axes[i].axis('off')  # Turn off axis labels for cleaner display

    plt.suptitle("Sample MNIST Digits (Raw Pixel Values)", y=0.98)  # Overall title
    plt.tight_layout()
    plt.show()

    #Displaying from the normalized flattened data, but first, we need to reshape it back to 28x28 for display
    _, axes_norm = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        # Reshape the flattened, normalized image back to 28x28 for display
        # x_train[i] is a (784,) array, reshape it to (28, 28)
        axes_norm[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes_norm[i].set_title(f"Label: {train_labels[i]}")
        axes_norm[i].axis('off')

    plt.suptitle("Sample MNIST Digits (Normalized and Reshaped for Display)", y=0.98)
    plt.tight_layout()
    plt.show()


except FileNotFoundError:
    print("Error: Make sure the MNIST data files are in the same directory as your script.")
except ValueError as e:
    print(f"Error: {e}")