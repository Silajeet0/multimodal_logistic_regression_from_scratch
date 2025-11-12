import numpy as np

class MultiClassLogisticRegression:
    """
        An implementation of the Logistic
        Regression using only numpy and python

    """
    def __init__(self, n_features, n_classes):
        """Initializes the model parameters"""

        self.weight = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.loss_history = []
        self.val_loss_history = []

    def softmax(self, score_mtx):
        """Helper method to convert the linear weights to valid probabilities"""

        # 1. Numerical Stability: Subtract max score from each row
        # This prevents np.exp() from overflowing with large scores
        stable_z = score_mtx - np.max(score_mtx, axis=1, keepdims=True)

        # 2. Calculate the exponentials of the stable scores
        exp_scores = np.exp(stable_z)

        # 3. Sum the exponentials for each row (per-sample)
        # axis=1 sums along the rows, keepdims=True maintains the 2D
        # structure for broadcasting, using keepdims = True makes
        #sum_exp_scores a 2D array instead of 1D vector
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)

        # 4. Divide each row's exponentials by that row's sum
        probabilities = exp_scores / sum_exp_scores

        return probabilities

    def compute_loss(self, y, y_pred):
        """
            Categorical Cross-Entropy Loss also called Negative Log Likelihood
            here we use average instead of just the sum as minimizing the sum gives
            the exact same weights as minimizing the average, but averaging it out makes
            the learning rate stable and independent of the batch size.
        """
        return -np.sum(y * np.log(y_pred + 1e-15)) / y.shape[0]

    def fit(self, X, Y, learning_rate = 0.05, epochs = 500, X_val=None, Y_val=None, patience=20, min_delta=1e-4):
        """
            Trains the model using gradient descent.

            Args:
                X (np.array): Training data of shape (n_samples, n_features).
                y (np.array): True one-hot encoded labels of shape (n_samples, n_classes).
                learning_rate (float): The step size for gradient descent.
                epochs (int): The number of passes over the training data.
                learning_rate (float): The step size for gradient descent.
                epochs (int): The maximum number of passes over the training data.
                X_val (np.array, optional): Validation data of shape (n_val_samples, n_features).
                                        If provided, early stopping will be used.
                Y_val (np.array, optional): True one-hot encoded validation labels.
                patience (int, optional): Number of epochs to wait for improvement
                                         on the validation loss before stopping.
                min_delta (int, optional): The minimum decrease in validation loss required to count
                                           as an "improvement" and reset the patience counter.
                                           Defaults to 1e-4.
            Returns:
                The learned weights, bias, and loss history.
        """

        best_val_loss = np.inf
        patience_counter = 0
        best_weights = self.weight.copy()
        best_bias = self.bias.copy()
        n_samples, _ = X.shape

        print(f"Training with Early Stopping: patience={patience} epochs")

        for i in range(epochs):
            # Training Step
            linear_model = (X @ self.weight) + self.bias
            y_pred = self.softmax(linear_model)
            loss = self.compute_loss(Y, y_pred)
            self.loss_history.append(loss)

            dw = (1 / n_samples) * (X.T @ (y_pred - Y))
            db = (1 / n_samples) * np.sum(y_pred - Y, axis=0)

            self.weight -= learning_rate * dw
            self.bias -= learning_rate * db

            # Validation Step
            if X_val is not None and Y_val is not None:
                # Calculate validation loss
                val_linear_model = (X_val @ self.weight) + self.bias
                val_y_pred = self.softmax(val_linear_model)
                val_loss = self.compute_loss(Y_val, val_y_pred)
                self.val_loss_history.append(val_loss)

                if (i + 1) % 10 == 0: #to print the model training details at intervals of 10
                    print(f"Epoch {i + 1}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

                # Check for improvement
                if (best_val_loss - val_loss) > min_delta:
                    best_val_loss = val_loss
                    best_weights = self.weight.copy()  # Save the best weights
                    best_bias = self.bias.copy()
                    patience_counter = 0  # Reset patience
                else:
                    # No meaningful improvement
                    patience_counter += 1

                # 3. Check for early stopping
                if patience_counter >= patience:
                    print(f"\n--- Early stopping triggered at epoch {i + 1} ---")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    break
            else:
                # No validation set provided, just print training loss
                if (i + 1) % 100 == 0:
                    print(f"Epoch {i + 1}/{epochs}, Train Loss: {loss:.4f}")

        # Restore the best weights found during training
        self.weight = best_weights
        self.bias = best_bias

        return self

    def predict(self, X):
        """
            Makes predictions on new data using the learned parameters
        """
        linear_model = (X @ self.weight) + self.bias

        y_predicted_probabilities = self.softmax(linear_model)
        y_predicted_label = np.argmax(y_predicted_probabilities, axis = 1)

        return np.array(y_predicted_label)