import numpy as np

class LogisticRegression:
    """
        An implementation of the Logistic
        Regression using only numpy and python

    """
    def __init__(self, n_features):
        """Initializes the model parameters"""

        self.weight = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []

    def sigmoid(self, z):
        """Helper method to convert the linear weights to valid probabilities"""
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        """
            Binary Cross-Entropy Loss also called Negative Log Likelihood obtained from Bernoulli Distribution
            here we use np.mean in order to work with the average loss instead of the total loss
        """
        return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

    def fit(self, X, Y, learning_rate = 0.05, epochs = 500):
        """
            Trains the model using gradient descent.

            Args:
                X (np.array): Training data of shape (n_samples, n_features).
                y (np.array): True labels of shape (n_samples,).
                learning_rate (float): The step size for gradient descent.
                epochs (int): The number of passes over the training data.

            Returns:
                The learned weights, bias, and loss history.
        """

        n_samples, _ = X.shape

        for _ in range(epochs):
            linear_model = (X @ self.weight) + self.bias
            y_pred = self.sigmoid(linear_model)

            loss = self.compute_loss(Y, y_pred)
            self.loss_history.append(loss)

            dw = (1 / n_samples) * (X.T @ (y_pred - Y))
            db = (1 / n_samples) * np.sum(y_pred - Y)

            self.weight -= learning_rate * dw
            self.bias -= learning_rate * db

        return self.weight, self.bias, self.loss_history

    def predict(self, X):
        """
            Makes predictions on new data using the learned parameters
        """
        linear_model = np.dot(X, self.weight) + self.bias

        y_predicted_probabilities = self.sigmoid(linear_model)
        y_predicted_labels = [1 if i > 0.5 else 0 for i in y_predicted_probabilities]

        return np.array(y_predicted_labels)