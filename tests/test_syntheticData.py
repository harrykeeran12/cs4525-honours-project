import unittest
from pathlib import Path

# To run the tests, use python3 -m unittest tests/<name of this py file>.py


class TestSyntheticData(unittest.TestCase):
    def test_trainingDataset(self):
        """Test if the training dataset exists."""
        self.assertTrue(
            Path.exists(Path("datasets/training_data1.csv")),
            "Run the data manipulation script in order to download the training dataset.",
        )


if __name__ == "__main__":
    unittest.main()
