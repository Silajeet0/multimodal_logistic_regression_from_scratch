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

    def _softmax(self, score_mtx):
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

    def fit(self, X, Y, learning_rate = 0.05, epochs = 500):
        """
            Trains the model using gradient descent.

            Args:
                X (np.array): Training data of shape (n_samples, n_features).
                y (np.array): True one-hot encoded labels of shape (n_samples, n_classes).
                learning_rate (float): The step size for gradient descent.
                epochs (int): The number of passes over the training data.

            Returns:
                The learned weights, bias, and loss history.
        """

        n_samples, _ = X.shape

        for _ in range(epochs):
            linear_model = (X @ self.weight) + self.bias
            y_pred = self._softmax(linear_model)

            loss = self.compute_loss(Y, y_pred)
            self.loss_history.append(loss)

            dw = (1 / n_samples) * (X.T @ (y_pred - Y))
            db = (1 / n_samples) * np.sum(y_pred - Y, axis = 0)

            self.weight -= learning_rate * dw
            self.bias -= learning_rate * db

        return self.weight, self.bias, self.loss_history

    def predict(self, X):
        """
            Makes predictions on new data using the learned parameters
        """
        linear_model = (X @ self.weight) + self.bias

        y_predicted_probabilities = self._softmax(linear_model)
        y_predicted_label = np.argmax(y_predicted_probabilities, axis = 1)

        return np.array(y_predicted_label)