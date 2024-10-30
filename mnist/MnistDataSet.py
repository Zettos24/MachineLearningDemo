import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def relu(x):
    """Applies the ReLU activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array with ReLU applied.
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """Computes the derivative of the ReLU function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Derivative of the ReLU function.
    """
    return np.where(x > 0, 1, 0)


def softmax(x):
    """Applies the softmax activation function.

    Args:
        x (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Output array after applying softmax.
    """
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


class MnistDataset:
    """A class to load, preprocess, and train a neural network on the MNIST dataset.

    Attributes:
        x (numpy.ndarray): Feature data.
        y (numpy.ndarray): Target labels.
        x_train (numpy.ndarray): Training feature data.
        x_test (numpy.ndarray): Testing feature data.
        y_train (numpy.ndarray): Training target labels.
        y_test (numpy.ndarray): Testing target labels.
        y_train_encoded (numpy.ndarray): One-hot encoded training labels.
        y_test_encoded (numpy.ndarray): One-hot encoded testing labels.
        input_size (int): Number of input features (784 for MNIST).
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of output classes (10 for digits 0-9).
        learning_rate (float): Learning rate for the optimizer.
        w1 (numpy.ndarray): Weights for the hidden layer.
        b1 (numpy.ndarray): Biases for the hidden layer.
        w2 (numpy.ndarray): Weights for the output layer.
        b2 (numpy.ndarray): Biases for the output layer.
        epochs (int): Number of training epochs.
        train_accuracies (list): List to store training accuracies per epoch.
        test_accuracies (list): List to store test accuracies per epoch.
        test_predictions (numpy.ndarray): Predictions on the test set.
    """

    def __init__(self):
        """Initializes the dataset, preprocesses data, and initializes network parameters."""
        # Step 1: Download the MNIST dataset
        mnist = fetch_openml('mnist_784', version=1)
        self.x, self.y = mnist.data, mnist.target.astype(np.int64)

        # Convert x to a numpy array and normalize the data
        self.x = np.array(self.x) / 255.0

        # Train and test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )

        # One-hot encoding of labels
        encoder = OneHotEncoder(sparse_output=False)
        # Convert y_train and y_test to numpy arrays before reshaping
        self.y_train_encoded = encoder.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_test_encoded = encoder.transform(self.y_test.values.reshape(-1, 1))

        # Initialize network parameters
        self.input_size = 784  # 28x28 images
        self.hidden_size = 128  # Number of neurons in the hidden layer
        self.output_size = 10  # 10 classes (digits 0-9)
        self.learning_rate = 0.01  # Learning rate

        # Initialize weights and biases
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1. / self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.output_size)
        self.b2 = np.zeros((1, self.output_size))

        # Initialize results
        self.epochs = 5
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_predictions = None

    def forward_pass(self, x):
        """Performs the forward pass through the network.

        Args:
            x (numpy.ndarray): Input data for the forward pass.

        Returns:
            tuple: Tuple containing the output of the network and the activations of the hidden layer.
        """
        z1 = np.dot(x, self.w1) + self.b1
        a1 = relu(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = softmax(z2)
        return a2, a1  # Return a1 for use in backward pass

    def backward_pass(self, x, y_true, output, a1):
        """Performs the backward pass and updates the weights and biases.

        Args:
            x (numpy.ndarray): Input data.
            y_true (numpy.ndarray): True labels.
            output (numpy.ndarray): Network output.
            a1 (numpy.ndarray): Activations from the hidden layer.
        """
        m = y_true.shape[0]

        # Output layer error
        dz2 = output - y_true
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer error
        dz1 = np.dot(dz2, self.w2.T) * relu_derivative(a1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m

        # Update parameters
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def train(self):
        """Trains the model on the MNIST dataset.

        Returns:
            tuple: Tuple containing test accuracies and training accuracies for each epoch.
        """
        # Training the model
        for epoch in range(self.epochs):
            output, a1 = self.forward_pass(self.x_train)
            self.backward_pass(self.x_train, self.y_train_encoded, output, a1)

            # Calculate training accuracy
            train_predictions = np.argmax(output, axis=1)
            train_acc = accuracy_score(self.y_train, train_predictions)
            self.train_accuracies.append(train_acc)

            # Calculate test accuracy
            test_output, _ = self.forward_pass(self.x_test)
            self.test_predictions = np.argmax(test_output, axis=1)
            test_acc = accuracy_score(self.y_test, self.test_predictions)
            self.test_accuracies.append(test_acc)

            print(f"Epoch {epoch + 1}/{self.epochs}, Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

        return self.test_accuracies, self.train_accuracies

    def reset_results(self):
        """Resets the training and testing accuracy lists and predictions."""
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_predictions = None

    def visualize_training(self):
        """Visualizes the training results, including accuracies and a confusion matrix.

        Also shows misclassified examples from the test set.
        """
        # Confusion matrix for the test data
        conf_matrix = confusion_matrix(self.y_test, self.test_predictions)

        # Plotting accuracy over epochs
        plt.figure(figsize=(14, 6))

        # Plot of training and test accuracy
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.epochs + 1), self.train_accuracies, label='Training Accuracy', marker='o')
        plt.plot(range(1, self.epochs + 1), self.test_accuracies, label='Test Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        # Confusion matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False,
                    xticklabels=np.arange(10), yticklabels=np.arange(10))
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')

        # Show misclassified examples
        misclassified = np.where(self.test_predictions != self.y_test)[0]
        num_examples = min(5, len(misclassified))  # Show up to 5 misclassifications

        plt.figure(figsize=(10, 3))
        for i in range(num_examples):
            index = misclassified[i]
            plt.subplot(1, num_examples, i + 1)
            plt.imshow(self.x_test[index].reshape(28, 28), cmap='gray')

            # Accessing the value in the pandas Series using .iloc
            plt.title(f"True: {self.y_test.iloc[index]}, Pred: {self.test_predictions[index]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()
