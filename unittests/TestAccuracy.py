import unittest

from mnist.MnistDataSet import MnistDataset


class MNISTAccuracyTEST(unittest.TestCase):
    """Unit test class for evaluating the accuracy of the MNIST dataset model."""

    mnist_dataset = None

    @classmethod
    def setUpClass(cls):
        """Set up the MNIST dataset before any tests are run.

        Initializes the MnistDataset instance, which downloads and prepares
        the MNIST dataset for training and testing.
        """
        cls.mnist_dataset = MnistDataset(10)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are completed.

        Resets the results of the MnistDataset instance to ensure no
        residual data affects subsequent tests.
        """
        cls.mnist_dataset.reset_results()

    def test_accuracy(self):
        """Test the accuracy of the model.

        Runs the training process and checks that the final training and
        test accuracies are greater than 0. Raises AssertionError if
        the conditions are not met.
        """
        test_accuracies, train_accuracies = self.mnist_dataset.train()

        # Assert that the training accuracy is greater than 0
        self.assertGreater(train_accuracies[-1], 0, "Training accuracy should be greater than 0%")

        # Assert that the test accuracy is greater than 0
        self.assertGreater(test_accuracies[-1], 0, "Test accuracy should be greater than 0%")

    def test_visualization(self):
        """Test the visualization of training results.

        Calls the visualize_training method to ensure that it executes
        without errors. This method will plot the accuracy over epochs
        and display the confusion matrix.
        """
        self.mnist_dataset.visualize_training()


if __name__ == '__main__':
    unittest.main()