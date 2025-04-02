import unittest
from pathlib import Path

# To run the tests, use python3 -m unittest tests/<name of this py file>.py


class TestEvaluateData(unittest.TestCase):
    def test_testingDataset(self):
        """Test if the testing dataset exists."""
        self.assertTrue(
            Path.exists(Path("datasets/testing_data.csv")),
            "Please check the GitHub repository or the provided dissertation for the data. Alternatively, contact the owner by email.",
        )

    def test_annotated(self):
        """Test if the annotated version of the testing dataset exists."""
        self.assertTrue(
            Path.exists(Path("datasets/annotated_testing_data.csv")),
            "Please check the GitHub repository or the provided dissertation for the data. Alternatively, contact the owner by email.",
        )

    def test_prompt1_mistral(self):
        """Test if the file for prompt 1 of mistral exists."""
        self.assertTrue(
            Path.exists(Path("datasets/mistral_latest_inference.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt1_qwen(self):
        """Test if the file for prompt 1 of qwen exists."""
        self.assertTrue(
            Path.exists(Path("datasets/qwen2.5_latest_inference.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt1_falcon(self):
        """Test if the file for prompt 1 of falcon exists."""
        self.assertTrue(
            Path.exists(Path("datasets/falcon3_latest_inference.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt2_mistral(self):
        """Test if the file for prompt 2 of mistral exists."""
        self.assertTrue(
            Path.exists(Path("datasets/mistral_latest_inference_prompt2.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt2_qwen(self):
        """Test if the file for prompt 2 of qwen exists."""
        self.assertTrue(
            Path.exists(Path("datasets/qwen2.5_latest_inference_prompt2.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt2_falcon(self):
        """Test if the file for prompt 2 of falcon exists."""
        self.assertTrue(
            Path.exists(Path("datasets/falcon3_latest_inference_prompt2.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt3_mistral(self):
        """Test if the file for prompt 3 of mistral exists."""
        self.assertTrue(
            Path.exists(Path("datasets/mistral_latest_inference_prompt3.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt3_qwen(self):
        """Test if the file for prompt 3 of qwen exists."""
        self.assertTrue(
            Path.exists(Path("datasets/qwen2.5_latest_inference_prompt3.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt3_falcon(self):
        """Test if the file for prompt 3 of falcon exists."""
        self.assertTrue(
            Path.exists(Path("datasets/falcon3_latest_inference_prompt3.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )


if __name__ == "__main__":
    unittest.main()
