import numpy as np
import matplotlib.pyplot as plt
from getData import readMNIST
from model import MultiClassLogisticRegression
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

train_label_path = "Dataset/fashionmnist/train-labels"
train_image_path = "Dataset/fashionmnist/train-images"
test_label_path = "Dataset/fashionmnist/t10k-labels"
test_image_path = "Dataset/fashionmnist/t10k-images"

# One-hot-encode method to create a binary array to handle categorical classes
#where only the correct class has a corresponding value as 1 the rest are 0
def one_hot_encode(labels, n_classes):
    one_hot_labels = np.zeros((labels.size, n_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels

# --- Main script execution ---
try:

    train_images, train_labels = readMNIST(label_filepath=train_label_path,
                                           image_filepath=train_image_path)

    test_images, test_labels = readMNIST(label_filepath=test_label_path,
                                         image_filepath=test_image_path)

    print(f"\nData loaded successfully! Shapes: {train_images.shape, train_labels.shape}")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # 5. PREPROCESSING
    print("Preprocessing data...")

    # a. Flatten Images
    x_train = train_images.reshape(train_images.shape[0], -1)
    x_test = test_images.reshape(test_images.shape[0], -1)

    # b. Normalize Pixels
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"X data preprocessed. Shape: {x_train.shape, x_test.shape}")

    # c. One-Hot Encode Training Labels
    y_train_one_hot = one_hot_encode(train_labels, 10)

    print(f"Y training labels one-hot encoded. Shape: {y_train_one_hot.shape}")
    print("\n--- Data successfully loaded and preprocessed! ---")

    # 6. TRAIN THE MODEL
    print("Starting model training...")
    n_features = x_train.shape[1]
    n_classes = 10

    model = MultiClassLogisticRegression(n_features=n_features, n_classes=n_classes)
    model.fit(x_train, y_train_one_hot, learning_rate=0.1, epochs=500)
    print("Training complete.")

    # 7. EVALUATE THE MODEL
    print("Evaluating model...")
    y_predictions = model.predict(x_test)
    accuracy = np.mean(y_predictions == test_labels)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    # This report gives you Precision, Recall, and F1-score for each class
    target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(classification_report(test_labels, y_predictions, target_names=target_names))

    # --- Confusion Matrix ---
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(test_labels, y_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 5. PLOT LOSS
    plt.figure()  # Create a new figure
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