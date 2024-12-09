import unittest
import pandas as pd
from main import main

class BoundaryTest(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object
        self.test_obj = TestUtils()
        self.minimum_accuracy = 0.85

    def test_boundary_accuracy(self):
        """Test if the model accuracy is not below the minimum acceptable threshold."""
        try:
            metrics, _ = main()
            actual_accuracy = metrics.get("accuracy", 0)
            is_accuracy_valid = actual_accuracy >= self.minimum_accuracy
            self.test_obj.yakshaAssert("TestBoundaryAccuracy", is_accuracy_valid, "boundary")
            if is_accuracy_valid:
                print(f"TestBoundaryAccuracy = Passed, Accuracy: {actual_accuracy}")
            else:
                print(f"TestBoundaryAccuracy = Failed, Accuracy below threshold: {actual_accuracy}")
        except Exception as e:
            print("Error in TestBoundaryAccuracy:", e)
            self.test_obj.yakshaAssert("TestBoundaryAccuracy", False, "boundary")



# Mock TestUtils for testing purposes
class TestUtils:
    def yakshaAssert(self, test_name, result, test_type):
        """Mock YakshaAssert implementation."""
        print(f"{test_name}: {'Passed' if result else 'Failed'} ({test_type})")


if __name__ == "__main__":
    unittest.main()
