# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas evaluate mauve-text
from pprint import pprint
import pandas as pd
import json
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from schema import RadiologyErrors

sBERTModel = SentenceTransformer("all-MiniLM-L6-v2")

accuracy_metric = evaluate.load("accuracy")

evaluation_metric = evaluate.load("bertscore")

evalFile = "datasets/evaluationResults.csv"

similarityFile = "datasets/similarityResults.csv"

confusionFile = "datasets/confusionMatrix.csv"

THRESHOLD = 0.7

df = pd.read_csv("datasets/annotated_testing_data.csv")


def red(string: str):
    return f"\033[31m{string}\033[0m"


def green(string: str):
    return f"\033[32m{string}\033[0m"


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


def JSONtoErrorArray(jsonString: RadiologyErrors):
    """Write function to turn the JSON strings into a count of the errors."""
    errorArray = [0, 0, 0, 0]
    radioError = json.loads(jsonString.model_dump_json())
    for i in radioError["errorsForWholeText"]:
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

    otherDF["JSON"] = otherDF[COLUMN_NAME].apply(
        lambda t: RadiologyErrors.model_validate_json(t)
    )
    # print(otherDF["JSON"][0])

    otherDF["Error Array"] = otherDF["JSON"].apply(lambda s: JSONtoErrorArray(s))

    # For all the errors in errorsForWholeText, get the explanations

    otherDF["Error Explanations"] = otherDF["JSON"].apply(
        lambda t: "\n".join(["".join(s.errorExplanation) for s in t.errorsForWholeText])
    )

    otherDF["Embeddings"] = otherDF["Error Explanations"].apply(
        lambda e: sBERTModel.encode(e)
    )

    otherDF["Total Errors"] = otherDF["JSON"].apply(lambda s: len(s.errorsForWholeText))

    errorIsolation(otherDF)

    otherDF.columns


def dataFrameSimilarity(df1: pd.DataFrame, df2: pd.DataFrame):
    """Uses the SBERT model to calculate sentence similarity of the error explanations."""
    if len(df1) != len(df2):
        raise Exception("Dataframes are not of the same size.")
    else:
        length = len(df1)
        arr = []
        for i in range(length):
            # Encode the value:
            # print(df1["Error Explanations"][i], df2["Error Explanations"][i])
            embed1 = sBERTModel.encode(df1["Error Explanations"][i])
            embed2 = sBERTModel.encode(df2["Error Explanations"][i])
            sim = sBERTModel.similarity(embed1, embed2)
            arr.append(sim.item())
        return pd.Series(arr)


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
    }
    return evaluationResults


def dataFrameComparison(otherDF: pd.DataFrame):
    """Compares a dataframe with the reference checking if they identified the correct errors, and had similar explanations(calculated by SBERT) depending on a specific threshold. Returns the confusion matrix."""

    confusionMatrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}

    TPs = []
    FNs = []
    print(len(df["Decoded"]))
    for index in range(len(otherDF["JSON"])):
        # Get all radiology errors.
        predictedList = otherDF["JSON"][index].errorsForWholeText
        referenceList = df["Decoded"][index].errorsForWholeText
        # print(predictedExplanations)

        for referenceError in referenceList:
            referenceErrorPhrase = " ".join(referenceError.errorPhrases)
            simRef = sBERTModel.encode(referenceErrorPhrase)
            print(
                f"({index}) {green(referenceErrorPhrase)}\n {green(referenceError.errorExplanation)}"
            )
            for predictedError in predictedList:
                predictedErrorPhrase = " ".join(predictedError.errorPhrases)
                print(
                    f"\t - (\n{red(predictedErrorPhrase)} \n{red(predictedError.errorExplanation)} \n {f'Similarity: {sBERTModel.similarity(simRef, sBERTModel.encode(predictedErrorPhrase))}'}"
                )
            prompt = input("Does a match exist? TP, FN or FP: ")
            if prompt == "TP":
                TPs.append(referenceError)
                confusionMatrix["TP"] += 1
            elif prompt == "FN":
                FNs.append(referenceError)
                confusionMatrix["FN"] += 1
            elif prompt == "FP":
                confusionMatrix["FP"] += 1
            else:
                pass

    return confusionMatrix


def main():
    # Decode the JSON strings.
    df["Decoded"] = df["Reference JSON Output"].apply(
        lambda x: RadiologyErrors.model_validate_json(cleanJSON(x))
    )
    df["Error Array"] = df["Decoded"].apply(lambda s: JSONtoErrorArray(s))

    df["Counted I"] = df["Error Array"].apply(lambda ea: ea["Internal Inconsistency"])
    df["Counted O"] = df["Error Array"].apply(lambda ea: ea["Omission"])
    df["Counted T"] = df["Error Array"].apply(lambda ea: ea["Transcription Error"])
    df["Counted E"] = df["Error Array"].apply(lambda ea: ea["Extraneous Statement"])

    df["Error Explanations"] = df["Decoded"].apply(
        lambda t: "\n".join(["".join(s.errorExplanation) for s in t.errorsForWholeText])
    )

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

    m2df = pd.read_csv("datasets/mistral_latest_inference_prompt2.csv")

    dataCorrection("mistral:latest", m2df)

    q2df = pd.read_csv("datasets/qwen2.5_latest_inference_prompt2.csv")

    dataCorrection("qwen2.5:latest", q2df)

    f2df = pd.read_csv("datasets/falcon3_latest_inference_prompt2.csv")

    dataCorrection("falcon3:latest", f2df)

    m3df = pd.read_csv("datasets/mistral_latest_inference_prompt3.csv")

    dataCorrection("mistral:latest", m3df)

    q3df = pd.read_csv("datasets/qwen2.5_latest_inference.csv")

    dataCorrection("qwen2.5:latest", q3df)

    f3df = pd.read_csv("datasets/falcon3_latest_inference.csv")

    dataCorrection("falcon3:latest", f3df)

    if not Path.exists(Path(evalFile)):
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
                pd.DataFrame([dataEvaluation("Mistral Prompt 2", m2df)]),
                pd.DataFrame([dataEvaluation("Qwen Prompt 2", q2df)]),
                pd.DataFrame([dataEvaluation("Falcon Prompt 2", f2df)]),
                pd.DataFrame([dataEvaluation("Mistral Prompt 3", m3df)]),
                pd.DataFrame([dataEvaluation("Qwen Prompt 3", q3df)]),
                pd.DataFrame([dataEvaluation("Falcon Prompt 3", f3df)]),
            ],
            ignore_index=True,
        )

        print(evaluationDataFrame)

        evaluationDataFrame.to_csv(evalFile)
    else:
        print(
            f"Evaluation file already exists. Please delete the file located at {evalFile} and run this program again."
        )
    if not Path.exists(Path(similarityFile)):
        similarityDataFrame = pd.DataFrame(
            {
                "Mistral Prompt 1": dataFrameSimilarity(df, mdf),
                "Qwen Prompt 1": dataFrameSimilarity(df, qdf),
                "Falcon Prompt 1": dataFrameSimilarity(df, fdf),
                "Mistral Prompt 2": dataFrameSimilarity(df, m2df),
                "Qwen Prompt 2": dataFrameSimilarity(df, q2df),
                "Falcon Prompt 2": dataFrameSimilarity(df, f2df),
                "Mistral Prompt 3": dataFrameSimilarity(df, m3df),
                "Qwen Prompt 3": dataFrameSimilarity(df, q3df),
                "Falcon Prompt 3": dataFrameSimilarity(df, f3df),
            },
        )

        print(similarityDataFrame)

        similarityDataFrame.to_csv(similarityFile)
    else:
        print(
            f"Similarity dataset already exists. Please delete the file located at {similarityFile} and run this program again."
        )
    if not Path.exists(Path(confusionFile)):
        confusionMatrix = pd.DataFrame(
            {
                # "Mistral Prompt 1": dataFrameComparison(mdf),
                # "Qwen Prompt 1": dataFrameComparison(qdf),
                # "Falcon Prompt 1": dataFrameComparison(fdf),
                # "Mistral Prompt 2": dataFrameComparison(m2df),
                # "Qwen Prompt 2": dataFrameComparison(q2df),
                # "Falcon Prompt 2": dataFrameComparison(f2df),
                "Mistral Prompt 3": dataFrameComparison(m3df),
                # "Qwen Prompt 3": dataFrameComparison(q3df),
                # "Falcon Prompt 3": dataFrameComparison(f3df),
            },
        )

        print(confusionMatrix)

        confusionMatrix.to_csv(confusionFile)
    else:
        print(
            f"Confusion Matrix data already exists. Please delete {confusionFile} and run this again."
        )


if __name__ == "__main__":
    main()
