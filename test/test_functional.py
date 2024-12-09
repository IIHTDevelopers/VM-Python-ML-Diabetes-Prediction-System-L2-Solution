import unittest
import os
from main import main

class FunctionalTest(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object
        self.test_obj = TestUtils()

        # Expected accuracy
        self.expected_accuracy = 0.97

        # Expected predictions for specific individuals
        self.expected_predictions = {
            "John Doe": "Non-Diabetic",
            "Jane Smith": "Non-Diabetic",
            "Alice Johnson": "Diabetic",
            "Bob Brown": "Diabetic",
            "Charlie Davis": "Non-Diabetic",
            "Emily White": "Non-Diabetic",
            "Frank Harris": "Non-Diabetic",
            "Grace Lee": "Non-Diabetic",
            "Hannah Young": "Diabetic",
            "Ian Thomas": "Non-Diabetic",
        }

    def test_model_file_exists(self):
        """Test if the model file exists in the directory."""
        model_file_path = "diabetes_model.pkl"
        file_exists = os.path.exists(model_file_path)
        self.test_obj.yakshaAssert("TestModelFileExists", file_exists, "functional")
        if file_exists:
            print("TestModelFileExists = Passed")
        else:
            print("TestModelFileExists = Failed")

    def test_preprocessor_file_exists(self):
        """Test if the preprocessor file exists in the directory."""
        preprocessor_file_path = "preprocessor.pkl"
        file_exists = os.path.exists(preprocessor_file_path)
        self.test_obj.yakshaAssert("TestPreprocessorFileExists", file_exists, "functional")
        if file_exists:
            print("TestPreprocessorFileExists = Passed")
        else:
            print("TestPreprocessorFileExists = Failed")

    def test_model_accuracy(self):
        """Test if the model achieves the expected accuracy."""
        try:
            metrics, _ = main()
            actual_accuracy = metrics.get("accuracy", 0)
            is_accuracy_correct = abs(actual_accuracy - self.expected_accuracy) < 1e-2
            self.test_obj.yakshaAssert("TestModelAccuracy", is_accuracy_correct, "functional")
            if is_accuracy_correct:
                print("TestModelAccuracy = Passed")
            else:
                print(f"Expected Accuracy: {self.expected_accuracy}, Actual Accuracy: {actual_accuracy}")
                print("TestModelAccuracy = Failed")
        except Exception as e:
            print("Error in TestModelAccuracy:", e)
            self.test_obj.yakshaAssert("TestModelAccuracy", False, "functional")
            print("TestModelAccuracy = Failed")

    def test_predictions(self):
        """Test if predictions for individuals match expected values."""
        try:
            _, predictions = main()
            mismatches = [
                (key, self.expected_predictions[key], predictions.get(key, None))
                for key in self.expected_predictions
                if predictions.get(key, None) != self.expected_predictions[key]
            ]

            if not mismatches:
                self.test_obj.yakshaAssert("TestPredictions", True, "functional")
                print("TestPredictions = Passed")
            else:
                print("Mismatches found:")
                for name, expected, actual in mismatches:
                    print(f"Name: {name}, Expected: {expected}, Actual: {actual}")
                self.test_obj.yakshaAssert("TestPredictions", False, "functional")
                print("TestPredictions = Failed")
        except Exception as e:
            print("Error in TestPredictions:", e)
            self.test_obj.yakshaAssert("TestPredictions", False, "functional")
            print("TestPredictions = Failed")


# Mock TestUtils for testing purposes
class TestUtils:
    def yakshaAssert(self, test_name, result, test_type):
        """Mock YakshaAssert implementation."""
        print(f"{test_name}: {'Passed' if result else 'Failed'} ({test_type})")


if __name__ == "__main__":
    unittest.main()
