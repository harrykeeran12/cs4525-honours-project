import unittest
from pathlib import Path
from evaluateData import JSONtoErrorArray, cleanJSON
import pandas as pd
from schema import RadiologyErrors
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

    def test_annotated_errorNum(self):
        df = pd.read_csv("datasets/annotated_testing_data.csv")
        df["Decoded"] = df["Reference JSON Output"].apply(
            lambda x: RadiologyErrors.model_validate_json(cleanJSON(x))
        )
        df["Error Array"] = df["Decoded"].apply(lambda s: JSONtoErrorArray(s))

        df["Counted I"] = df["Error Array"].apply(
            lambda ea: ea["Internal Inconsistency"]
        )
        df["Counted O"] = df["Error Array"].apply(lambda ea: ea["Omission"])
        df["Counted T"] = df["Error Array"].apply(lambda ea: ea["Transcription Error"])
        df["Counted E"] = df["Error Array"].apply(lambda ea: ea["Extraneous Statement"])
        # print("Results in Reference Data vs Counted Reference Data Results: ")

        # print(
        #     f"Omissions: {sum(df['Omission'])}: Number of counted omissions in testing data: {sum(df['Counted O'])}"
        # )

        # print(
        #     f"Internal Inconsistency: {sum(df['Internal Inconsistency'])}Number of counted internal inconsistencies in testing data: {sum(df['Counted I'])}"
        # )
        # print(
        #     f"Extraneous Statements: {sum(df['Extraneous Statement'])} Number of counted extraneous statements in testing data: {sum(df['Counted E'])}"
        # )
        # print(
        #     f"Transcription Errors: {sum(df['Transcription Error'])}Number of counted transcription errors in testing data: {sum(df['Counted T'])}"
        # )

        # df["Omission"] = df["Counted O"]
        # df["Extraneous Statement"] = df["Counted E"]
        # df["Internal Inconsistency"] = df["Counted I"]
        # df["Transcription Error"] = df["Counted T"]
        # df.to_csv("datasets/annotated_testing_data.csv")

        # print(sum(df['Omission'] + df["Extraneous Statement"] + df["Internal Inconsistency"] + df["Transcription Error"]))

        self.assertEqual(
            sum(df["Omission"]),
            sum(df["Counted O"]),
            msg="The number of omission errors reported are inconsistent with the JSON.",
        )
        self.assertEqual(
            sum(df["Internal Inconsistency"]),
            sum(df["Counted I"]),
            msg="The number of internal inconsistency errors reported are inconsistent with the JSON.",
        )
        self.assertEqual(
            sum(df["Extraneous Statement"]),
            sum(df["Counted E"]),
            msg="The number of extraneous statements errors reported are inconsistent with the JSON.",
        )
        self.assertEqual(
            sum(df["Transcription Error"]),
            sum(df["Counted T"]),
            msg="The number of transcription errors reported are inconsistent with the JSON.",
        )

    def test_prompt1_mistral(self):
        """Test if the file for prompt 1 of mistral exists."""
        self.assertTrue(
            Path.exists(Path("datasets/inference/mistral_latest_inference.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt1_qwen(self):
        """Test if the file for prompt 1 of qwen exists."""
        self.assertTrue(
            Path.exists(Path("datasets/inference/qwen2.5_latest_inference.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt1_falcon(self):
        """Test if the file for prompt 1 of falcon exists."""
        self.assertTrue(
            Path.exists(Path("datasets/inference/falcon3_latest_inference.csv")),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt2_mistral(self):
        """Test if the file for prompt 2 of mistral exists."""
        self.assertTrue(
            Path.exists(
                Path("datasets/inference/mistral_latest_inference_prompt2.csv")
            ),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt2_qwen(self):
        """Test if the file for prompt 2 of qwen exists."""
        self.assertTrue(
            Path.exists(
                Path("datasets/inference/qwen2.5_latest_inference_prompt2.csv")
            ),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt2_falcon(self):
        """Test if the file for prompt 2 of falcon exists."""
        self.assertTrue(
            Path.exists(
                Path("datasets/inference/falcon3_latest_inference_prompt2.csv")
            ),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt3_mistral(self):
        """Test if the file for prompt 3 of mistral exists."""
        self.assertTrue(
            Path.exists(
                Path("datasets/inference/mistral_latest_inference_prompt3.csv")
            ),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt3_qwen(self):
        """Test if the file for prompt 3 of qwen exists."""
        self.assertTrue(
            Path.exists(
                Path("datasets/inference/qwen2.5_latest_inference_prompt3.csv")
            ),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )

    def test_prompt3_falcon(self):
        """Test if the file for prompt 3 of falcon exists."""
        self.assertTrue(
            Path.exists(
                Path("datasets/inference/falcon3_latest_inference_prompt3.csv")
            ),
            "Please run the prelim_eval.py file, or consult the README.md .",
        )


if __name__ == "__main__":
    unittest.main()
