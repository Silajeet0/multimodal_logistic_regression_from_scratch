import numpy as np
import matplotlib.pyplot as plt
import time  # To time our models
from getData import readMNIST
from model import MultiClassLogisticRegression

# --- Scikit-learn Imports ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- File Paths ---
train_label_path = "Dataset/fashionmnist/train-labels"
train_image_path = "Dataset/fashionmnist/train-images"
test_label_path = "Dataset/fashionmnist/t10k-labels"
test_image_path = "Dataset/fashionmnist/t10k-images"


# --- Helper Function ---
def one_hot_encode(labels, n_classes):
    """
    Converts a 1D array of labels into a 2D one-hot encoded matrix.
    """
    one_hot_labels = np.zeros((labels.size, n_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels


# --- Main script execution ---
try:
    # 1. LOAD DATA
    train_images, train_labels = readMNIST(label_filepath=train_label_path,
                                           image_filepath=train_image_path)
    test_images, test_labels = readMNIST(label_filepath=test_label_path,
                                         image_filepath=test_image_path)
    print(f"\nData loaded successfully! Shapes: {train_images.shape, train_labels.shape}")

    # 2. PREPROCESSING
    print("Preprocessing data...")
    # a. Flatten Images
    x_train = train_images.reshape(train_images.shape[0], -1)  # Shape -> (60000, 784)
    x_test = test_images.reshape(test_images.shape[0], -1)  # Shape -> (10000, 784)

    # b. Normalize Pixels
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print(f"X data preprocessed. Shape: {x_train.shape, x_test.shape}")

    # c. One-Hot Encode Training Labels (for our model only)
    y_train_one_hot = one_hot_encode(train_labels, 10)
    print(f"Y training labels one-hot encoded. Shape: {y_train_one_hot.shape}")
    print("\n--- Data successfully loaded and preprocessed! ---")

    # 3. TRAIN & EVALUATE YOUR FROM-SCRATCH MODEL
    print("\n--- Training From-Scratch Model ---")
    n_features = x_train.shape[1]  # 784
    n_classes = 10

    start_time = time.time()

    model = MultiClassLogisticRegression(n_features=n_features, n_classes=n_classes)
    model.fit(x_train, y_train_one_hot, learning_rate=0.1, epochs=1000)

    end_time = time.time()
    from_scratch_time = end_time - start_time

    print(f"Training complete in {from_scratch_time:.2f} seconds.")

    print("Evaluating from-scratch model...")
    y_pred_scratch = model.predict(x_test)
    accuracy_scratch = np.mean(y_pred_scratch == test_labels)

    # Store results for final comparison
    results = {
        "From-Scratch LR": {
            "accuracy": accuracy_scratch,
            "time": from_scratch_time
        }
    }

    # 4. TRAIN & EVALUATE SKLEARN MODELS
    print("\n--- Training Scikit-learn Models (The Bake-Off) ---")

    # Define the models to compare
    models_to_compare = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Linear SVM": LinearSVC(max_iter=2000, dual=True),  # dual=True is often faster
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "RBF Kernel SVM": SVC(kernel='rbf', C=1.0)  # C=1.0 is a reasonable default
    }

    # Loop, train, time, and evaluate each model
    for name, sk_model in models_to_compare.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        # Train the model. Sklearn models take the original 1D labels.
        sk_model.fit(x_train, train_labels)

        end_time = time.time()
        train_time = end_time - start_time

        # Evaluate
        y_pred_sk = sk_model.predict(x_test)
        accuracy_sk = np.mean(y_pred_sk == test_labels)

        print(f"Training complete in {train_time:.2f} seconds.")
        print(f"Accuracy: {accuracy_sk * 100:.2f}%")

        # Store results
        results[name] = {"accuracy": accuracy_sk, "time": train_time}

    # 5. FINAL RESULTS SUMMARY
    print("\n\n--- FINAL PROJECT BAKE-OFF RESULTS ---")
    print("-----------------------------------------------------")
    print(f"| {'Model Name':<25} | {'Test Accuracy':<15} | {'Training Time (s)':<18} |")
    print("|" + "-" * 26 + "|" + "-" * 17 + "|" + "-" * 20 + "|")

    for name, metrics in results.items():
        print(f"| {name:<25} | {metrics['accuracy'] * 100:<14.2f}% | {metrics['time']:<18.2f} |")
    print("-----------------------------------------------------")

    # 6. DETAILED REPORT FOR YOUR FROM-SCRATCH MODEL
    print("\n\n--- Detailed Report for From-Scratch Model ---")
    print(f"Accuracy: {results['From-Scratch LR']['accuracy'] * 100:.2f}%")

    target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred_scratch, target_names=target_names))

    # --- Confusion Matrix ---
    print("Generating Confusion Matrix for From-Scratch Model...")
    cm = confusion_matrix(test_labels, y_pred_scratch)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('From-Scratch Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # --- Loss Curve ---
    plt.figure()  # Create a new figure
    plt.plot(model.loss_history)
    plt.title("From-Scratch Model Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.show()


except FileNotFoundError:
    print("\n--- ERROR: FileNotFoundError ---")
    print("Could not find the data files. Check your paths.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")