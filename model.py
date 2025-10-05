import numpy as np

class LogisticRegression:
    def __init__(self, nFeatures):
        self.weight = np.zeros(nFeatures)
        self.bias = 0
        self.lossHistory = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        # Binary Cross-Entropy Loss
        return -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))

    def gradientDescent(self, X, Y, learningRate = 0.05, epochs = 500):
        nSamples, nFeatures = X.shape

        for _ in range(epochs):
            linearModel = (X @ self.weight) + self.bias
            yPred = self.sigmoid(linearModel)

            loss = self.compute_loss(Y, yPred)
            self.lossHistory.append(loss)

            dw = (1 / nSamples) * (X.T @ (yPred - Y))
            db = (1 / nSamples) * np.sum(yPred - Y)

            self.weight -= learningRate * dw
            self.bias -= learningRate * db

        return self.weight, self.bias, self.lossHistory