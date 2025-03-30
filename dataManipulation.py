from huggingface_hub import whoami, hf_hub_download
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

user = whoami(token=os.getenv("HF_TOKEN"))
print(user["name"])

REPO_ID = "ibrahimhamamci/CT-RATE"
FILENAME = "dataset/radiology_text_reports/train_reports.csv"

RANDOM_SEED = 42

CT_RATE = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
)
# Remove the duplicates and the VolumeName: we just want the reports.
CT_RATE.drop_duplicates(inplace=True)
CT_RATE.drop(labels="VolumeName", axis=1, inplace=True)
CT_RATE_LENGTH = CT_RATE.shape[0]
# display(CT_RATE)

"""Wrangling the data:"""

# For each row, put the information into one paragraph.

consolidatedList = []
for rowNumber in range(CT_RATE_LENGTH):
    # Get row:
    row = CT_RATE.iloc[rowNumber]
    consolidatedList.append(
        f"""Clinical Information:\n{row[0]}\nTechnique:\n{row[1]}\n Findings:\n{row[2]}\nImpressions: \n{row[3]}"""
    )
# These entries do not have any errors.
newDF = pd.DataFrame(
    {"Correct Items": consolidatedList, "Errors": [False for i in consolidatedList]}
)

newDF.drop_duplicates(inplace=True)

newDF = newDF.sample(frac=1).reset_index(drop=True)

outputfile = "training_data1.csv"
# print(f"Creating a new file called: {outputfile} with reports from {REPO_ID}.")
newDF.to_csv(f"datasets/{outputfile}")
# display(newDF)
