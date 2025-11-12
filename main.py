import numpy as np
import matplotlib.pyplot as plt
from getData import readMNIST
from model import MultiClassLogisticRegression
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import download_data

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "Dataset", "fashionmnist")

train_label_path = os.path.join(DATA_PATH, "train-labels")
train_image_path = os.path.join(DATA_PATH, "train-images")
test_label_path = os.path.join(DATA_PATH, "t10k-labels")
test_image_path = os.path.join(DATA_PATH, "t10k-images")

# AUTO-DOWNLOADER
# Check if the data exists. If not, download it.
if not os.path.exists(train_image_path):
    print("Data not found. Running downloader...")
    download_data.download_and_unzip()
else:
    print("Data found. Skipping download.")

# One-hot-encode method to create a binary array to handle categorical classes
# where only the correct class has a corresponding value as 1 while the rest are 0
def one_hot_encode(labels, n_classes):
    one_hot_labels = np.zeros((labels.size, n_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels

# Main script execution
try:

    train_images, train_labels = readMNIST(label_filepath=train_label_path,
                                           image_filepath=train_image_path)

    test_images, test_labels = readMNIST(label_filepath=test_label_path,
                                         image_filepath=test_image_path)

    print(f"\nData loaded successfully! Shapes: {train_images.shape, train_labels.shape}")

    # PREPROCESSING
    print("Preprocessing data...")

    # a. Flatten Images
    x_train = train_images.reshape(train_images.shape[0], -1)
    x_test = test_images.reshape(test_images.shape[0], -1)

    # b. Normalize Pixels
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # CREATE TRAIN, TEST AND VALIDATION SPLITS
    # We split the 60,000 images into 50,000 for training and 10,000 for validation
    # test_size=10000 means 10,000 samples go to the validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, train_labels, test_size=10000, random_state=42, stratify=train_labels)
    print(f"Training data split: {x_train.shape} (train), {x_val.shape} (validation)")

    # c. One-Hot Encode Training and Validation Labels
    y_train_one_hot = one_hot_encode(y_train, 10)
    y_val_one_hot = one_hot_encode(y_val, 10)  # <-- Need this for validation loss
    print(f"Y labels one-hot encoded.")
    print("\n--- Data successfully loaded and preprocessed! ---")

    # 6. TRAIN THE MODEL
    print("Starting model training...")
    n_features = x_train.shape[1]
    n_classes = 10

    model = MultiClassLogisticRegression(n_features=n_features, n_classes=n_classes)
    model.fit(
        x_train, y_train_one_hot,
        learning_rate=0.2,
        epochs=2000, # this was just used to demonstrate the early stopping feature of the model, from the analysis of
        # the loss curve, an epoch value of 500 works fairly well (which is used as the default value in the fit method)
        X_val=x_val,
        Y_val=y_val_one_hot,
        patience=45, # the model continues to train if the improvement is more than a certain threshold after 45 epochs, else it stops
        min_delta = 1e-3 # the threshold for early stopping
    )
    print("Training complete.")

    # EVALUATE THE MODEL
    print("Evaluating model...")

    # Training Set
    y_train_pred = model.predict(x_train)
    train_acc = np.mean(y_train_pred == y_train)
    train_err = 1.0 - train_acc
    print(f"\nTraining Accuracy:   {train_acc * 100:.2f}%")
    print(f"Training Error:   {train_err * 100:.2f}%")

    # Test Set
    y_test_pred = model.predict(x_test)
    test_acc = np.mean(y_test_pred == test_labels)
    test_err = 1.0 - test_acc
    print(f"Test Accuracy:       {test_acc * 100:.2f}%")
    print(f"Test Error:          {test_err * 100:.2f}%")

    print("\nClassification Report:")
    # This report gives us the Precision, Recall, and F1-score for each class
    target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(classification_report(test_labels, y_test_pred, target_names=target_names))

    # Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(test_labels, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # PLOT LOSS
    plt.figure()
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