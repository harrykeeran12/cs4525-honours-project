# Code for the preliminary evaluation section of my project.
# To run the application, `ollama` must be installed on the system. The `ollama` daemon must be run in the background using `ollama serve`.

import ollama
import pandas as pd
from pprint import pprint
from pathlib import Path
import logging
import datetime
import argparse
from schema import RadiologyError


CMDPARSER = argparse.ArgumentParser(
    prog="python3 honours_project/prelim_eval.py",
    description="This program generates an analysis of a report, looking for any errors using the help of a model specified. The default models expected to be installed on the machine are mistral:latest, falcon3:latest and qwen2.5:latest.",
    epilog="",
)

PWD = str(Path.cwd()) + "/honours_project"

# Create and configure logger.
logging.basicConfig(
    filename=PWD + f"/logs/prelim_eval_{datetime.datetime.now()}.log",
    format="%(asctime)s: %(levelname)s: %(message)s",
    filemode="w",
    level=logging.DEBUG,
)
# Configure the argument parser.
CMDPARSER.add_argument(
    "modelName",
    help="Name of the model used when evaluating a report, utilising ollama.",
)
ARGUMENTS = CMDPARSER.parse_args()
modelName = ARGUMENTS.modelName

# Creating an object
logging.debug("Starting logger.")

MODELSUSED = ["mistral:latest", "falcon3:latest", "qwen2.5:latest"]

try:
    installedModels = [model.get("model") for model in ollama.list().models]
    if len(installedModels) == 0:
        logging.error("No models installed on the system.")
        raise Exception(
            "There are no models installed on the system. Please install models using ollama pull <model-name>."
        )
except ConnectionError:
    raise Exception("Start Ollama with ollama serve in another terminal.")


# Get list of models names on the system.
for i in MODELSUSED:
    if i not in installedModels:
        raise Exception(
            f"Model {i} not on system. To install {i}, use ollama pull {i}."
        )
    else:
        logging.debug(f"Models on system: {i}")


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
if Path.exists(Path(PWD + "/datasets/preliminary_eval_results.csv")):
    # print("This path exists.")
    WRITEBYTE = "a"
    prelimEvalDF = pd.read_csv(PWD + "/datasets/preliminary_eval_results.csv")

else:
    # print("This path does not exist. Create this file.")
    WRITEBYTE = "w"
    logging.debug("Created new dataset.")
    # Feed each report into the models.

    # TODO: Remove the head method
    temp = removedCorrection.head(1)

    reportDict = {
        "Original report": temp,
        "mistral:latest": [""" """ for i in temp],
        "falcon3:latest": [""" """ for i in temp],
        "qwen2.5:latest": [""" """ for i in temp],
    }

    prelimEvalDF = pd.DataFrame(reportDict)

logging.debug(f"Dataframe loaded with shape {prelimEvalDF.shape}")

# Ollama keeps models in memory, so better to have the for-loop as a report for each model.


def createReportIssues(row: str, MODELNAME: str):
    """A function that creates a report using the ollama.generate function, taking in the name and the row. This allows it to be used with the dataframe.apply function."""
    logging.debug(f"{MODELNAME}: Finding errors")
    response = ollama.generate(
        model=MODELNAME,
        system=SYSTEM,
        prompt=row,
        options={"temperature": 0},
        format=RadiologyError.model_json_schema(),
    )["response"]
    logging.debug(f"{MODELNAME}: Completed finding errors.")
    logging.info(f"{MODELNAME}: {response}")
    return response


if modelName in installedModels:
    print(f"Model name specified: {modelName}")
    try:
        prelimEvalDF[modelName] = removedCorrection.apply(
            lambda x: createReportIssues(x, modelName)
        )
    finally:
        prelimEvalDF.to_csv(
            PWD + f"/datasets/preliminary_eval_results_{modelName}.csv", mode="w"
        )
        print("Returned data to .csv file.")

else:
    logging.error(f"Model name {modelName} could not be found on system.")
    raise Exception(f"The model name specified: {modelName} is not on the system.")
