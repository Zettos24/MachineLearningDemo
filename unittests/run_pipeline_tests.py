import unittest
from unittests.TestAccuracy import MNISTAccuracyTEST

# Create a test suite that contains specific tests
suite = unittest.TestSuite()

# Add the specific test to be executed on the pipeline
suite.addTest(MNISTAccuracyTEST('test_accuracy'))

if __name__ == '__main__':
    """
    Executes the test suite that contains the specific tests.

    The test runner is used to display the results of the tests,
    including information about passed and failed tests.
    """
    runner = unittest.TextTestRunner()
    runner.run(suite)
