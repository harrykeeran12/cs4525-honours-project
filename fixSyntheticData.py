# This file takes in the existing synthetic data that Dr Zhang has provided and for each report, randomly modifies each word. The 4 errors are: omissions, internal inconsistencies, extraneous statements, transcription errors.

from typing import List
import pandas as pd
import random
from pprint import pprint
from schema import RadiologyError, ErrorType, RadiologyErrors

random.seed(42)


def red(string: str):
    return f"\033[31m{string}\033[0m"


dataframe = pd.read_csv("datasets/300ormore.csv")

itemsToChange = dataframe["Correct Items"]

print(f"Current number of data to change: {len(itemsToChange)}")

internalInconsistency = [
    ["anterior", "posterior"],
    ["medial", "lateral"],
    ["superior", "inferior"],
    [
        "anterolateral",
        "posterolateral",
        "supralateral",
        "infralateral",
        "anterosuperior",
        "posterosuperior",
        "anteroposterior",
    ],
    [
        "anteromedial",
        "posteromedial",
        "supramedial",
        "inframedial",
        "anteroinferior",
        "posteroinferior",
        "posteroanterior",
    ],
    [
        "anterior-lateral",
        "posterior-lateral",
        "superior-lateral",
        "inferior-lateral",
        "anterior-superior",
        "posterior-superior",
    ],
    [
        "anterior-medial",
        "posterior-medial",
        "superior-medial",
        "inferior-medial",
        "anterior-inferior",
        "posterior-inferior",
    ],
    ["anterior-posterior", "medial-lateral", "superior-inferior"],
    ["dorsal-ventral", "transverse", "craniocaudal"],
    ["cranial", "caudal"],
    ["hepatopedal", "hepatofugal"],
    ["dorsal", "ventral"],
    ["proximal", "distal"],
    ["long axis", "short axis"],
    ["peripheral", "central"],
    ["superficial", "deep"],
    ["metaphysis", "diaphysis", "epiphysis"],
    ["ascending", "descending"],
    ["increase", "decrease"],
    ["increased", "decreased"],
    ["basal", "apical"],
    ["hyperdense", "hypodense"],
    ["solid", "cystic"],
    ["dependent", "non-dependent"],
    ["upper", "lower"],
]

transcription = [
    ["abscess", "access", "assess"],
    ["achalasia", "atelectasis", "epistaxis"],
    ["adrenal", "renal"],
    ["alveolar", "valcular", "lobular", "tubular"],
    ["aneurysm", "anaplasia", "anemia"],
    ["anterolisthesis", "retrolisthesis", "spondylolisthesis"],
    ["ascites", "cystitis", "bursitis", "colitic"],
    ["aspiration", "eventration"],
    ["atheroma", "myxoma", "osteoma", "lipoma"],
    ["borderline", "baseline"],
    ["bronchiectasis", "bronchitis", "bronchiolitis", "bronchi"],
    ["bronchogenic", "bronchiolitic", "bronchoscopic"],
    ["bullous", "mucous"],
    ["calcified", "ossified", "classified"],
    ["carcinomatosis,sarcomatosis", "carcinosis", "sarcoidosis"],
    ["cm", "mm", "m"],
    ["consolidation", "accumulation", "congestion", "compaction", "obstruction"],
    ["consolidative", "accumulative", "congestive", "obstructive"],
    ["coronary", "coronal", "coronoid", "coracoid", "corneal"],
    ["corpuscles", "corpus", "corvus", "corpse"],
    ["cortical", "corticoid", "corticate"],
    ["craniocaudal", "craniocervical", "craniobasal"],
    ["cyst", "gist", "list", "fist"],
    ["cystic", "systolic", "caustic", "cyclic", "plastic"],
    ["degenerative", "regenerative", "destructive"],
    ["diaphragm", "diagram", "diaphysis"],
    ["edematous", "erythematous", "emphysematous"],
    ["effusion", "confusion", "diffusion", "perfusion", "occlusion"],
    ["empyema", "emphysema", "haematoma", "endothelium"],
    ["endobronchial", "endotracheal"],
    ["esophagogastric", "esophagocolic"],
    ["esophagus", "esophagitis"],
    [
        "fibrosis",
        "stenosis",
        "sclerosis",
        "synostosis",
        "cyanosis",
        "thrombosis",
        "necrosis",
        "nephrosis",
        "silicosis",
        "cirrhosis",
        "asbestosis",
        "aspergillosis",
        "kyphosis",
        "lordosis",
        "mycosis",
    ],
    [
        "fibrotic",
        "stenotic",
        "sclerotic",
        "cyanotic",
        "thrombotic",
        "necrotic",
        "nephrotic",
        "cirrhotic",
    ],
    ["fissure", "fixture", "fisher", "fistula", "fossa"],
    ["fluid", "flutter", "fluctuant", "florid"],
    ["fracture", "friction", "contracture", "rapture"],
    ["fusiform", "reniform"],
    ["gastroesophageal", "gastroduodenal", "gastrojejunal", "gastroepiploic"],
    ["ground glass/ground-glass", "ground grass", "brown glass", "brown brass"],
    ["hemorrhagic", "hemostatic", "hemolytic"],
    ["hernia", "fistula", "myalgia"],
    ["herniation", "fistulation"],
    ["hilar", "hyoid", "hilum"],
    [
        "hypertension",
        "hypotension",
        "hyperextension",
        "hyperattenuation",
        "hypoattenuation",
    ],
    ["indeterminant", "intermittent"],
    ["inflammatory", "informatory", "inspiratory"],
    ["intrapulmonary", "intraperitoneal", "intramedullary", "intravascular"],
    ["lobular", "lobar"],
    ["lymphangitis", "pancreatitis", "adenitis"],
    ["lymphatic", "hepatic"],
    ["marrow", "narrow", "macro", "micro"],
    ["medullary", "modular"],
    ["metastasis", "metaphysis", "metanalysis", "metastases"],
    ["metastatic", "metaplastic", "myoclonic", "metabolic", "hyperplastic"],
    ["millimetric", "metric"],
    ["myocardial", "myocardium", "endocardial", "endocardium", "pericardium"],
    ["nodule", "module", "tuber"],
    ["non-specific", "non-systemic", "non-selective"],
    ["occlusive", "conclusive", "inclusive"],
    ["osteopenia", "sarcopenia"],
    ["osteopenic", "osteoporotic", "osteolytic", "osteopathic"],
    ["paratracheal", "paraoesophageal", "parabronchial", "pericardial"],
    ["parenchyma", "pneumonia"],
    ["pathological", "physiological", "psychological"],
    [
        "pericarditis",
        "endocarditis",
        "pleuritis",
        "perichondritis",
        "peritonitis",
        "pneumonitis",
    ],
    ["perivascular", "perihilar", "peribronchial"],
    ["plaque", "black", "plug"],
    ["pneumothorax", "hemothorax"],
    ["portal", "total", "pedal"],
    ["previous", "pervious", "pylorus"],
    ["pulmonary", "voluntary"],
    ["reticular", "auricular", "trabecular", "vesicular", "articular"],
    [
        "reticulation",
        "recirculation",
        "recalculation",
        "regulation",
        "strangulation",
        "ventilation",
        "speculation",
        "stipulation",
    ],
    ["retropulsion", "retroversion", "retroflexion", "reflexion", "expulsion"],
    ["sequela", "sclera", "stella"],
    ["sequelae", "sequestrae"],
    ["significant", "malignant", "magnificant", "consistent"],
    ["subphrenic", "subpleural", "subhepatic", "subdural"],
    ["suspicious", "surreptitious"],
    ["traction", "fraction", "action", "contraction", "reaction"],
    ["vascular", "valvular", "muscular", "vestibular", "molecular"],
    ["vocal", "focal", "vagal", "local"],
    ["lymphadenopathy", "adenopathy", "radiculopathy"],
    ["pleurodesis", "pleurocentesis"],
]

omission = [
    "no",
    "cannot",
    "clear",
    "clearly",
    "exclude",
    "excluded",
    "increase",
    "decrease",
    "significant",
    "more",
    "greater",
    "less",
]

extraneous = [
    "the total",
    "quina",
    "management",
    "office",
    "staircase",
    "hesitation",
    "umbrella",
    "keyboard",
    "carriage",
]

# Convert side confusion and near_homonym into dictionaries.
sideConfusionDict = {}

for mistakeWords in internalInconsistency:
    for word in mistakeWords:
        # Create a set from the mistake words
        mistakeSet = set(mistakeWords)
        wordSet = {word}
        # print(f"Current word = {wordSet} - {mistakeSet - wordSet}")
        sideConfusionDict[word] = mistakeSet - wordSet

nearHomonymDict = {}

for homonyms in transcription:
    #   print(homonyms)
    for currentHomonym in homonyms:
        # print(currentHomonym)
        closeHomonymSet = set(homonyms)
        # print(f"Current word = {currentHomonym} - {closeHomonymSet - {currentHomonym}}")
        nearHomonymDict[currentHomonym] = closeHomonymSet - {currentHomonym}

# pprint(sideConfusionDict)
# pprint(nearHomonymDict)

errorTupleList = []
errorJSONList: List[RadiologyErrors] = []
for item in itemsToChange:
    # Split the items by spaces
    splitItem = item.split(" ")
    jsonList: RadiologyErrors = []
    # Each item has an error array, which keeps track of any errors added.
    errorArray = [0, 0, 0, 0]
    for wordIndex in range(len(splitItem)):
        word = splitItem[wordIndex]
        flip = random.randrange(0, 4)
        if flip == 0:
            # Internal Inconsistency
            if word in sideConfusionDict:
                possibleWords = sideConfusionDict[word]
                randomIndex = random.randrange(0, len(possibleWords))
                replacementWord = list(possibleWords)[randomIndex]
                # print(f"---\nReplacing {word} -> {replacementWord}")
                # splitItem[wordIndex] = red(f"{replacementWord}({word})")
                jsonList.append(
                    RadiologyError(
                        errorType=ErrorType.InternalInconsistency,
                        errorPhrases=[f"{word}"],
                        errorExplanation=[
                            f"There is a side confusion as it should be {replacementWord}, instead of {word}"
                        ],
                    ).model_dump_json()
                )
                splitItem[wordIndex] = f"{replacementWord}"
                errorArray[0] += 1
                # print(word)
        elif flip == 1:
            # Transcription Errors
            if word in nearHomonymDict:
                homonyms = nearHomonymDict[word]
                if len(homonyms) == 1:
                    similarWord = list(homonyms)[0]
                else:
                    randomIndex = random.randrange(0, len(homonyms))
                    similarWord = list(homonyms)[randomIndex]
                # print(f"---\nReplacing {word} -> {similarWord}")
                jsonList.append(
                    RadiologyError(
                        errorType=ErrorType.TranscriptionError,
                        errorPhrases=[f"{word}"],
                        errorExplanation=[
                            f"There is a transcription error as it should be {word}, instead of {similarWord}"
                        ],
                    ).model_dump_json()
                )
                splitItem[wordIndex] = f"{similarWord}"
                errorArray[2] += 1
        elif flip == 2:
            # Omission
            if word in omission:
                splitItem[wordIndex] = ""
        elif flip == 3:
            # Extraneous statement
            pass
    errorJSONList.append("\n".join(jsonList))
    errorEntry = " ".join(splitItem)
    # print(f"Error count = {sum(errorArray)}")
    errorTupleList.append((errorEntry, errorArray))

newDf = pd.DataFrame(
    {
        "Original Report": [i for i in itemsToChange],
        "Reports with Errors": [i[0] for i in errorTupleList],
        "Errors": True,
    }
)

errorDf = pd.DataFrame(
    data=[i[1] for i in errorTupleList],
    columns=[
        "Internal Inconsistency",
        "Omission",
        "Transcription Error",
        "Extraneous Statement",
    ],
)


jsonDF = pd.DataFrame(data=errorJSONList)

# print(jsonDF)

jsonDF.to_csv("datasets/json.csv")

print(f"Length of error list: {len(errorJSONList)}")
syntheticData = pd.concat([newDf, errorDf, jsonDF], axis=1)

syntheticData.to_csv("datasets/syntheticData.csv")
