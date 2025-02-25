# Code for the preliminary evaluation section of my project.
# To run the application, `ollama` must be installed on the system. The `ollama` daemon must be run in the background using `ollama serve`.

import ollama
import pandas as pd
from pprint import pprint
from pathlib import Path
import logging

# Create and configure logger
prelim_logger = logging.basicConfig(
    filename="./honours_project/prelim_eval.log",
    format="%(asctime)s: %(levelname)s: %(message)s",
    filemode="w",
    level=logging.DEBUG,
)

# Creating an object

prelim_logger.debug("Starting logger.")

MODELSUSED = ["mistral:latest", "falcon3:latest", "qwen2.5:latest"]

modelNames = [model.get("model") for model in ollama.list().models]

# Get list of models names on the system.
print(f"Models used: {MODELSUSED}")
for i in MODELSUSED:
    if i not in modelNames:
        raise Exception(f"Model {i} not on system.")
    logging.debug(f"Found model: {i}")


SYSTEM = """You help correct radiology report errors. These include transcription errors, internal inconsistencies, insertion statements and translation errors. For each mistake, show the incorrect words and explain what the problem is."""

PWD = str(Path.cwd()) + "/honours_project"

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
    # print(f"{i} : {errorNumber} errors.")
# display(errors)

# %%
# Feed each report into the models.
temp = removedCorrection
reportDict: dict[str, list[str]] = {
    "Original report": temp,
    "mistral:latest": [],
    "falcon3:latest": [],
    "qwen2.5:latest": [],
}

for report in temp:
    for name in MODELSUSED:
        generated = ollama.generate(
            name, prompt=SYSTEM + report, options={"temperature": 0}
        )
        reportDict[name].append(generated["response"])
        logging.debug(f"Generated new response to report {len(reportDict[name])} using model {name}.")

pprint(reportDict)


# Convert dictionary into CSV file.

tempData = pd.DataFrame().from_dict(reportDict)
tempData.to_csv(PWD + "datasets/preliminary_eval_results.csv")
logging.debug("Created new datasets.")

# display(tempData)
