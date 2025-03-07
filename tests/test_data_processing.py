import sys
import os
import unittest
import pandas as pd

# Ensure the correct path is added
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import load_data, clean_text

class TestDataProcessing(unittest.TestCase):

    def test_load_data_csv(self):
        """Test loading a CSV file."""
        test_file = "tests/sample.csv"
        df = pd.DataFrame({"text": ["Hello World!", "Testing NLP."]})
        df.to_csv(test_file, index=False)

        loaded_df = load_data(test_file)
        self.assertEqual(len(loaded_df), 2)
        self.assertIn("text", loaded_df.columns)

        os.remove(test_file)  # Cleanup

    def test_load_data_txt(self):
        """Test loading a plain text file."""
        test_file = "tests/sample.txt"
        with open(test_file, "w") as f:
            f.write("This is a test.")

        loaded_text = load_data(test_file)
        self.assertEqual(loaded_text, ["This is a test."])

        os.remove(test_file)  # Cleanup

    def test_clean_text(self):
        """Test text preprocessing."""
        raw_text = "Hello, World! This is a TEST."
        expected_output = "hello world test"
        
        processed_text = clean_text(raw_text)
        self.assertEqual(processed_text, expected_output)

if __name__ == "__main__":
    unittest.main()
