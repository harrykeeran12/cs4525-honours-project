from flask import Flask, make_response, render_template, request, jsonify
import ollama
from pydantic import BaseModel
import json


class RadiologyError(BaseModel):
    """This class serves as a schema to act as as structured output for the models."""

    errorType: list[str]
    errorPhrases: list[str]
    errorExplanation: list[str]


app = Flask(__name__)

try:
    modelList = [model.get("model") for model in ollama.list().models]
except Exception:
    raise Exception("Ollama not running. Please start ollama.")


@app.route("/")
def index():
    return render_template("index.html", models=modelList)


@app.post("/generate")
def generate():
    """Takes in the model name + report information as a form. Outputs the possible errors, as JSON."""

    SYSTEM = "You help correct radiology report errors. These include transcription errors, internal inconsistencies, insertion statements and translation errors. For every mistake found in the text, show the incorrect words and explain what the problem is. The errorPhrases array must be the same length as the errorType array and errorExplanation array. Ignore any errors deemed unnecessary or redundant."

    SYSTEM2 = """You help correct radiology report errors. When provided with a free-text radiology report, analyze it for the following error types:
    - Transcription errors: Mistakes in spelling, punctuation, or word choice likely caused during dictation or transcription.
    - Internal inconsistencies: Contradictory statements within the same report.
    - Insertion statements: Text that appears to be incorrectly inserted or doesn't belong in the report.
    - Translation errors: Errors that occur when medical terminology is incorrectly used or interpreted.

    For each error found, identify:
    1. The exact text containing the error
    2. The specific type of error
    3. A clear explanation of why it's an error

    Return your analysis in the following JSON format:
    {
        "errorPhrases": "exact text with error",
        "errorType": "one of the four error types",
        "errorExplanation": "explanation of the problem",
    }

    The arrays must be consistent (each error needs all three elements). Ignore any errors deemed clinically insignificant or redundant. This system operates in an isolated local environment with no external connections.
    """

    modelName = request.form["modelName"]
    reportInfo = request.form["reportInformation"]
    if modelName in modelList:
        # return make_response(jsonify({"message": "Model name found."}), 200)
        # Ollama generates here:
        ollamaResponse = ollama.generate(
            model=modelName,
            system=SYSTEM2,
            prompt=reportInfo,
            keep_alive=0,
            options={"temperature": 0},
            format=RadiologyError.model_json_schema(),
        )["response"]

        listOfErrors = []

        jsonResponse = json.loads(ollamaResponse)
        print(jsonResponse)
        for errorNumber in range(len(jsonResponse["errorPhrases"])):
            errorPhrase = jsonResponse["errorPhrases"][errorNumber]
            errorDescription = jsonResponse["errorExplanation"][errorNumber]
            reportInfo = reportInfo.replace("\n", "</br>")
            reportInfo = reportInfo.replace(errorPhrase, f"<mark> {errorPhrase} </mark>")
            listOfErrors.append((errorPhrase, errorDescription))
        htmlResponse = render_template(
            "generatedResponse.html",
            correctedOutput=reportInfo,
            listOfErrors=listOfErrors,
        )

        return make_response(
            htmlResponse,
            201,
        )
    else:
        return make_response(
            jsonify({"error": f"Model name {modelName} not present in ollama models."}),
            404,
        )


if __name__ == "__main__":
    app.run(debug=True)
