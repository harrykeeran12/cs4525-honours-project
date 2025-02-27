# Code for the preliminary evaluation section of my project.
# To run the application, `ollama` must be installed on the system. The `ollama` daemon must be run in the background using `ollama serve`.

import ollama
import pandas as pd
from pprint import pprint
from pathlib import Path
import logging

PWD = str(Path.cwd()) + "/honours_project"

# Create and configure logger.
logging.basicConfig(
    filename=PWD + "/prelim_eval.log",
    format="%(asctime)s: %(levelname)s: %(message)s",
    filemode="w",
    level=logging.DEBUG,
)


# Creating an object

logging.debug("Starting logger.")

MODELSUSED = ["mistral:latest", "falcon3:latest", "qwen2.5:latest"]

modelNames = [model.get("model") for model in ollama.list().models]

# Get list of models names on the system.
print(f"Models used: {MODELSUSED}")
for i in MODELSUSED:
    if i not in modelNames:
        raise Exception(f"Model {i} not on system.")
    logging.debug(f"Found model: {i}")


SYSTEM = """You help correct radiology report errors. These include transcription errors, internal inconsistencies, insertion statements and translation errors. For each mistake, show the incorrect words and explain what the problem is."""



dataframe = pd.read_csv(PWD + "/datasets/testing_data.csv")

removedCorrection = dataframe["Removed Correction"]

correctedData = dataframe["Re-dictated"]

errors = dataframe["Type of Error"]
# Sanity check.
for i in range(len(removedCorrection)):
    if removedCorrection[i] == correctedData[i]:
        raise ValueError(
            f"Correction not removed properly in {i} : \n{removedCorrection[i]}"
        )

for i in range(len(errors)):
    errorList = errors[i].split(";")
    errorNumber = len(errorList)

# Check if file for preliminary evaluation already exists.

# Feed each report into the models.
temp = removedCorrection

reportDict: dict[str, list[str]] = {
    "Original report": temp,
    "mistral:latest": [""" """ for i in temp],
    "falcon3:latest": [""" """ for i in temp],
    "qwen2.5:latest": [""" """ for i in temp],
}

# Ollama keeps models in memory, so better to have the for-loop as a report for each model.


def createReportIssues(row: str, MODELNAME: str):
    """A function that creates a report using the ollama.generate function, taking in the name and the row. This allows it to be used with the dataframe.apply function."""
    logging.debug(f"{MODELNAME}: Finding errors")
    response = ollama.generate(MODELNAME, prompt=SYSTEM + row, options={"temperature": 0})["response"]
    reportDict[MODELNAME].append(response)
    logging.debug(f"{MODELNAME}: Completed finding errors.")
    logging.info(f"{MODELNAME}: {response}")
    return response

temp.apply(lambda x : createReportIssues(x, "mistral:latest"))
temp.apply(lambda x : createReportIssues(x, "falcon3:latest"))
temp.apply(lambda x : createReportIssues(x, "qwen2.5:latest"))


# Convert dictionary into CSV file.

tempData = pd.DataFrame().from_dict(reportDict)

tempData.to_csv(PWD + "/datasets/preliminary_eval_results.csv")

logging.debug("Created new dataset.")

# display(tempData)
