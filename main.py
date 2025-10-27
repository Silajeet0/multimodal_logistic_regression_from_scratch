import numpy as np
import matplotlib.pyplot as plt
from getData import readMNIST
from model import MultiClassLogisticRegression
import os  # Import the OS module

# --- 1. Get the directory this script is in ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory is: {SCRIPT_DIR}")

# --- 2. Define the subfolders where the data is ---
DATA_FOLDER = "Dataset"
SUB_FOLDER = "fashionmnist"

# --- 3. Build the correct, absolute paths ---
# This joins /home/.../ML_project + Dataset + fashionmnist + filename
train_label_path = os.path.join(SCRIPT_DIR, DATA_FOLDER, SUB_FOLDER, "train-labels")
train_image_path = os.path.join(SCRIPT_DIR, DATA_FOLDER, SUB_FOLDER, "train-images")
test_label_path = os.path.join(SCRIPT_DIR, DATA_FOLDER, SUB_FOLDER, "t10k-labels")
test_image_path = os.path.join(SCRIPT_DIR, DATA_FOLDER, SUB_FOLDER, "t10k-images")


# (Your one_hot_encode function)
def one_hot_encode(labels, n_classes):
    one_hot_labels = np.zeros((labels.size, n_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels


# --- Main script execution ---
try:
    # 4. LOAD DATA using the correct paths
    print("Loading data from:")
    print(f"  - {train_label_path}")
    print(f"  - {train_image_path}")

    train_images, train_labels = readMNIST(label_filepath=train_label_path,
                                           image_filepath=train_image_path)

    test_images, test_labels = readMNIST(label_filepath=test_label_path,
                                         image_filepath=test_image_path)

    print(f"\nData loaded successfully! Shapes: {train_images.shape, train_labels.shape}")

    # 5. PREPROCESSING
    print("Preprocessing data...")

    # a. Flatten Images
    x_train = train_images.reshape(train_images.shape[0], -1)
    x_test = test_images.reshape(test_images.shape[0], -1)

    # b. Normalize Pixels
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # c. Add Bias Term
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    print(f"X data preprocessed. Shape: {x_train.shape, x_test.shape}")

    # d. One-Hot Encode Training Labels
    y_train_one_hot = one_hot_encode(train_labels, 10)

    print(f"Y training labels one-hot encoded. Shape: {y_train_one_hot.shape}")
    print("\n--- Data successfully loaded and preprocessed! ---")

    # 6. (NEXT STEP) TRAIN YOUR MODEL
    print("Starting model training...")
    n_features = x_train.shape[1]  # Should be 785
    n_classes = 10

    model = MultiClassLogisticRegression(n_features=n_features, n_classes=n_classes)
    model.fit(x_train, y_train_one_hot, learning_rate=0.1, epochs=500)
    print("Training complete.")

    # 7. (NEXT STEP) EVALUATE YOUR MODEL
    print("Evaluating model...")
    y_predictions = model.predict(x_test)
    accuracy = np.mean(y_predictions == test_labels)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    # 8. (NEXT STEP) PLOT LOSS
    plt.plot(model.loss_history)
    plt.title("Model Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.show()


except FileNotFoundError:
    print("\n--- ERROR: FileNotFoundError ---")
    print("Could not find the data files at the expected path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(f"An unexpected error occurred: {e}")