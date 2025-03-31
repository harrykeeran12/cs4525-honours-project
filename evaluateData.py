# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas evaluate mauve-text
from pprint import pprint
import pandas as pd
import json
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer

sBERTModel = SentenceTransformer("all-MiniLM-L6-v2")

accuracy_metric = evaluate.load("accuracy")

evaluation_metric = evaluate.load("bertscore")

# Read the csv file
df = pd.read_csv("datasets/annotated_testing_data.csv")

print(f"Number of omissions in testing data: {sum(df['Omission'])}")

print(
    f"Number of internal inconsistencies in testing data: {sum(df['Internal Inconsistency'])}"
)
print(
    f"Number of extraneous statements in testing data: {sum(df['Extraneous Statement'])}"
)
print(
    f"Number of transcription errors in testing data: {sum(df['Transcription Error'])}"
)


def cleanJSON(s: str):
    """Unescape the JSON strings, as they have \n markers in them:"""
    decoded = bytes(s, "utf-8").decode("unicode_escape")
    return decoded


def errorIsolation(otherDF: pd.DataFrame):
    """Isolate the errors into their own columns from the JSON."""
    otherDF["Omission"] = otherDF["Error Array"].apply(lambda s: s["Omission"])

    otherDF["Internal Inconsistency"] = otherDF["Error Array"].apply(
        lambda s: s["Internal Inconsistency"]
    )

    otherDF["Transcription Error"] = otherDF["Error Array"].apply(
        lambda s: s["Transcription Error"]
    )

    otherDF["Extraneous Statement"] = otherDF["Error Array"].apply(
        lambda s: s["Extraneous Statement"]
    )


def JSONtoErrorArray(jsonString: str):
    """Write function to turn the JSON strings into a count of the errors."""
    errorArray = [0, 0, 0, 0]
    jsonString = json.loads(jsonString)
    for i in jsonString["errorsForWholeText"]:
        if i["errorType"] == "Internal Inconsistency":
            errorArray[0] += 1
        elif i["errorType"] == "Omission":
            errorArray[1] += 1
        elif i["errorType"] == "Transcription Error":
            errorArray[2] += 1
        elif i["errorType"] == "Extraneous Statement":
            errorArray[3] += 1
        else:
            pass
    return {
        "Internal Inconsistency": errorArray[0],
        "Omission": errorArray[1],
        "Transcription Error": errorArray[2],
        "Extraneous Statement": errorArray[3],
    }


def dataCorrection(COLUMN_NAME: str, otherDF: pd.DataFrame):
    """Takes in a dataframe and where the unescaped JSON strings are stored, and creates columns required for the evaluation."""

    if COLUMN_NAME not in otherDF.columns:
        raise ValueError(f"Column {COLUMN_NAME} not found in dataframe.")

    otherDF["JSON"] = otherDF[COLUMN_NAME].apply(lambda t: json.loads(t))

    otherDF["Error Array"] = otherDF[COLUMN_NAME].apply(lambda s: JSONtoErrorArray(s))

    # For all the errors in errorsForWholeText, get the explanations

    otherDF["Error Explanations"] = otherDF["JSON"].apply(
        lambda t: "\n".join(
            ["".join(s["errorExplanation"]) for s in t["errorsForWholeText"]]
        )
    )

    otherDF["Embeddings"] = otherDF["Error Explanations"].apply(
        lambda e: sBERTModel.encode(e)
    )

    otherDF["Total Errors"] = otherDF["JSON"].apply(
        lambda s: len(s["errorsForWholeText"])
    )

    errorIsolation(otherDF)

    otherDF.columns


def dataFrameSimilarity(df1: pd.DataFrame, df2: pd.DataFrame):
    """Uses the SBERT model to calculate sentence similarity of the error explanations."""
    if len(df1) != len(df2):
        raise Exception("Dataframes are not of the same size.")
    else:
        length = len(df1)
        arr = []
        # npa = np.empty(len(df1))
        for i in range(length):
            # Encode the value:
            # print(df1["Error Explanations"][i], df2["Error Explanations"][i])
            embed1 = sBERTModel.encode(df1["Error Explanations"][i])
            embed2 = sBERTModel.encode(df2["Error Explanations"][i])
            sim = sBERTModel.similarity(embed1, embed2)
            arr.append(sim[0])
            # np.append(npa, values=sim[0])
        print(arr)
        return np.mean(arr, dtype=np.float64)


def dataEvaluation(columnName: str, otherDF: pd.DataFrame):
    """Takes in a dataframe and evaluates the metrics."""
    total_accuracy = accuracy_metric.compute(
        predictions=otherDF["Total Errors"],
        references=df["Total Errors"],
        normalize=False,
    )

    omission = accuracy_metric.compute(
        predictions=otherDF["Omission"], references=df["Omission"], normalize=False
    )

    internal_inconsistency = accuracy_metric.compute(
        predictions=otherDF["Internal Inconsistency"],
        references=df["Internal Inconsistency"],
        normalize=False,
    )

    extraneous_statement = accuracy_metric.compute(
        predictions=otherDF["Extraneous Statement"],
        references=df["Extraneous Statement"],
        normalize=False,
    )

    transcription_error = accuracy_metric.compute(
        predictions=otherDF["Transcription Error"],
        references=df["Transcription Error"],
        normalize=False,
    )

    # Want to check the error explanations for semantic similaries.

    bertScore = evaluation_metric.compute(
        predictions=otherDF["Error Explanations"],
        references=df["Error Explanations"],
        lang="en",
        model_type="distilbert-base-uncased",
        use_fast_tokenizer=True,
    )

    # Average Explanation Similarity.

    print(f"Average SBERT Similarity = {dataFrameSimilarity(df, otherDF)}")

    evaluationResults: dict[str, float] = {
        "Name": columnName,
        "totalAccuracy": total_accuracy["accuracy"],
        "omission": omission["accuracy"],
        "internalInconsistency": internal_inconsistency["accuracy"],
        "extraneousStatement": extraneous_statement["accuracy"],
        "transcriptionError": transcription_error["accuracy"],
        "avg_BERT_precision": np.mean(bertScore["precision"]),
        "avg_BERT_recall": np.mean(bertScore["recall"]),
        "avg_BERT_f1": np.mean(bertScore["f1"]),
        "avg_similarity": dataFrameSimilarity(df, otherDF),
    }
    return evaluationResults


# Decode the strings.
df["Decoded"] = df["Reference JSON Output"].apply(lambda x: cleanJSON(x))

# Convert to JSON.
df["JSON Objects"] = df["Decoded"].apply(lambda s: json.loads(s))

df["Error Array"] = df["Decoded"].apply(lambda s: JSONtoErrorArray(s))

# print(df["Error Array"])

df["Error Explanations"] = df["JSON Objects"].apply(
    lambda t: "\n".join(
        ["".join(s["errorExplanation"]) for s in t["errorsForWholeText"]]
    )
)

df["Embeddings"] = df["Error Explanations"].apply(lambda e: sBERTModel.encode(e))

# Find the total number of errors in each JSON string.
df["Total Errors"] = df["Error Array"].apply(lambda s: sum(s.values()))

errorIsolation(df)

# Add the data generated by the models and clean them up.

mdf = pd.read_csv("datasets/mistral_latest_inference.csv")

dataCorrection("mistral:latest", mdf)

qdf = pd.read_csv("datasets/qwen2.5_latest_inference.csv")

dataCorrection("qwen2.5:latest", qdf)

fdf = pd.read_csv("datasets/falcon3_latest_inference.csv")

dataCorrection("falcon3:latest", fdf)

mtdf = pd.read_csv("datasets/mistral_latest_inference_prompt2.csv")

dataCorrection("mistral:latest", mtdf)

qtdf = pd.read_csv("datasets/qwen2.5_latest_inference_prompt2.csv")

dataCorrection("qwen2.5:latest", qtdf)

ftdf = pd.read_csv("datasets/falcon3_latest_inference_prompt2.csv")

dataCorrection("falcon3:latest", ftdf)

# Evaluation of data generated by prompt 1:

evaluationDataFrame = pd.DataFrame(
    {},
    columns=[
        "Name",
        "totalAccuracy",
        "omission",
        "internalInconsistency",
        "extraneousStatement",
        "transcriptionError",
        "avg_BERT_precision",
        "avg_BERT_recall",
        "avg_BERT_f1",
        "avg_similarity",
    ],
)
# print(f"{df['Error Explanations'][0]} \n\t\n {mtdf['Error Explanations'][0]}")

print("Concatenating dataframes.")

evaluationDataFrame = pd.concat(
    [
        evaluationDataFrame,
        pd.DataFrame([dataEvaluation("Mistral Prompt 1", mdf)]),
        pd.DataFrame([dataEvaluation("Qwen Prompt 1", qdf)]),
        pd.DataFrame([dataEvaluation("Falcon Prompt 1", fdf)]),
        pd.DataFrame([dataEvaluation("Mistral Prompt 2", mtdf)]),
        pd.DataFrame([dataEvaluation("Qwen Prompt 2", qtdf)]),
        pd.DataFrame([dataEvaluation("Falcon Prompt 2", ftdf)]),
    ],
    ignore_index=True,
)
# print("Mistral Evaluation Prompt 1 Evaluation")

# print(dataEvaluation("Mistral Prompt 1",mdf))

# print("Qwen Evaluation Prompt 1 Evaluation")

# print(dataEvaluation("Qwen Prompt 1",qdf))

# print("Falcon Evaluation Prompt 1 Evaluation")

# print(dataEvaluation("Falcon Prompt 1", fdf))

# Evaluation of data generated by prompt 2:

# print("Mistral Evaluation Prompt 2 Evaluation")

# print(dataEvaluation("Mistral Prompt 2",mtdf))

# print("Qwen Evaluation Prompt 2 Evaluation")

# print(dataEvaluation("Qwen Prompt 2",qtdf))

# print("Falcon Evaluation Prompt 2 Evaluation")

# print(dataEvaluation("Falcon Prompt 2",ftdf))

print(evaluationDataFrame)

evaluationDataFrame.to_csv("datasets/evaluationResults.csv")
