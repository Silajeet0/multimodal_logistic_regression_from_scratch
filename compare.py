import numpy as np
import matplotlib.pyplot as plt
import time  # To time our models
import pickle  # To save our model
from getData import readMNIST
from model import MultiClassLogisticRegression

# --- Scikit-learn Imports ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

# --- File Paths ---
train_label_path = "Dataset/fashionmnist/train-labels"
train_image_path = "Dataset/fashionmnist/train-images"
test_label_path = "Dataset/fashionmnist/t10k-labels"
test_image_path = "Dataset/fashionmnist/t10k-images"


# --- Helper Function ---
def one_hot_encode(labels, n_classes):
    one_hot_labels = np.zeros((labels.size, n_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels


# --- Plotting Function ---
def plot_comparison_chart(results, metric_name, title):
    """Helper function to plot a comparison bar chart for a given metric."""
    model_names = list(results.keys())
    metric_values = [res[metric_name] for res in results.values()]

    plt.figure(figsize=(10, 6))
    colors = ['#ff6666'] + ['#66b3ff'] * (len(model_names) - 1)  # Highlight your model
    bars = plt.bar(model_names, metric_values, color=colors)

    plt.ylabel(metric_name)
    plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 1.0)  # All these metrics are between 0 and 1

    # Add the value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval * 100:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


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
    x_train = train_images.reshape(train_images.shape[0], -1).astype('float32') / 255.0
    x_test = test_images.reshape(test_images.shape[0], -1).astype('float32') / 255.0
    print(f"X data preprocessed. Shape: {x_train.shape, x_test.shape}")

    y_train_one_hot = one_hot_encode(train_labels, 10)
    print(f"Y training labels one-hot encoded. Shape: {y_train_one_hot.shape}")
    print("\n--- Data successfully loaded and preprocessed! ---")

    # 3. TRAIN & EVALUATE THE HANDCRAFTED LOGISTIC REGRESSION MODEL
    print("\n--- Training From-Scratch Model ---")
    n_features = x_train.shape[1]  # 784
    n_classes = 10

    start_time = time.time()

    model = MultiClassLogisticRegression(n_features=n_features, n_classes=n_classes)
    # Using 500 epochs based on previous loss curve analysis
    model.fit(x_train, y_train_one_hot, learning_rate=0.1, epochs=500)

    end_time = time.time()
    from_scratch_time = end_time - start_time

    print(f"Training complete in {from_scratch_time:.2f} seconds.")

    print("Evaluating from-scratch model...")
    y_pred_scratch = model.predict(x_test)

    # Get all metrics
    report_scratch = classification_report(test_labels, y_pred_scratch, output_dict=True)

    results = {
        "From-Scratch LR": {
            "accuracy": report_scratch['accuracy'],
            "precision": report_scratch['weighted avg']['precision'],
            "recall": report_scratch['weighted avg']['recall'],
            "f1-score": report_scratch['weighted avg']['f1-score'],
            "time": from_scratch_time
        }
    }

    # 4. TRAIN & EVALUATE SKLEARN MODELS
    print("\n--- Training Scikit-learn Models ---")

    models_to_compare = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Linear SVM": LinearSVC(max_iter=2000, dual=True),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "RBF Kernel SVM": SVC(kernel='rbf', C=1.0)
    }

    for name, sk_model in models_to_compare.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        sk_model.fit(x_train, train_labels)
        end_time = time.time()
        train_time = end_time - start_time

        # Evaluate
        y_pred_sk = sk_model.predict(x_test)
        report_sk = classification_report(test_labels, y_pred_sk, output_dict=True)

        print(f"Training complete in {train_time:.2f} seconds.")

        # Store results
        results[name] = {
            "accuracy": report_sk['accuracy'],
            "precision": report_sk['weighted avg']['precision'],
            "recall": report_sk['weighted avg']['recall'],
            "f1-score": report_sk['weighted avg']['f1-score'],
            "time": train_time
        }

    # FINAL RESULTS SUMMARY (PLOTS)
    print("\nGenerating comparison charts...")
    plot_comparison_chart(results, "accuracy", "Model Accuracy Comparison")
    plot_comparison_chart(results, "precision", "Model Precision (Weighted Avg) Comparison")
    plot_comparison_chart(results, "recall", "Model Recall (Weighted Avg) Comparison")
    plot_comparison_chart(results, "f1-score", "Model F1-Score (Weighted Avg) Comparison")

    # --- Confusion Matrix ---
    target_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
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
    plt.title("From-Scratch Model Loss During Training (500 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.show()


except FileNotFoundError:
    print("\n--- ERROR: FileNotFoundError ---")
    print("Could not find the data files. Check your paths.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")