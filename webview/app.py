from pydantic import ValidationError
from ..schema import RadiologyErrors
from flask import Flask, make_response, render_template, request, jsonify
import ollama
import webbrowser
import json


app = Flask(__name__)
try:
    modelList = [model.get("model") for model in ollama.list().models]

except Exception:
    raise Exception("Ollama not running. Please start ollama.")
try:
    webbrowser.open("http://127.0.0.1:5000/")
except Exception as e:
    raise e("Unable to open a web browser.")


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
        "errorType": "an array containing the four error types",
        "errorPhrases": "exact text with error",
        "errorExplanation": "explanation of the problem",
    }

    The arrays must be consistent (each error needs all three elements). Ignore any errors deemed clinically insignificant or redundant. This system operates in an isolated local environment with no external connections.
    """
    SYSTEM3 = """
    Your task is to identify errors in unstructured radiology reports including omissions, extraneous statements, transcription errors, and internal inconsistencies. Analyze each report and output errors in JSON format.
    Example 1:
    Input: "Clinical Information:\nNot given.\nTechnique:\nNon-contrast images were taken in the axial plane with a section thickness of 1.5 m.\nFindings:\nOther findings are stable.\nImpressions: \nNot given."

    Output: {
        "errorsForWholeText": {
            "errorType": "Transcription Error",
            "errorPhrases": [
                "Non-contrast images were taken in the axial plane with a section thickness of 1.5 m."
            ],
            "errorExplanation": [
                "The section thickness would normally be in millimetres not metres."
            ]
        }
    }

    Example 2:
    Input: "Clinical Information:\nPatient with chronic headaches.\nTechnique:\nMRI of the brain without contrast.\nFindings:\nNo acute intracranial abnormality.\nNo evidence of mass effect or midline shift.\nVentricles are normal in size and configuration.\nImpressions:\nNormal brain MRI."

    Output: {
        "errorsForWholeText": "No errors found"
    }

    Example 3:
    Input: "Clinical Information:\nFall from standing height.\nTechnique:\nCT scan of the right wrist.\nFindings:\nThere is a comminuted fracture of the distal radius.\nNo evidence of dislocation.\nImpressions:\nThe patient has a sprained wrist."

    Output: {
        "errorsForWholeText": {
            "errorType": "Internal Inconsistency",
            "errorPhrases": [
                "There is a comminuted fracture of the distal radius.",
                "The patient has a sprained wrist."
            ],
            "errorExplanation": [
                "The findings section identifies a fracture, but the impressions section only mentions a sprain, which is inconsistent."
            ]
        }
    }

    Analyse the report below:
    """

    modelName = request.form["modelName"]
    reportInfo = request.form["reportInformation"]
    if modelName in modelList:
        # return make_response(jsonify({"message": "Model name found."}), 200)
        # Ollama generates here:
        ollamaResponse = ollama.generate(
            model=modelName,
            system=SYSTEM3,
            prompt=json.dumps(reportInfo),
            options={"temperature": 0},
            format=RadiologyErrors.model_json_schema(),
        )["response"]

        listOfErrors = []
        try:
            jsonResponses = RadiologyErrors.model_validate_json(
                ollamaResponse
            ).errorsForWholeText
        except ValidationError as ve:
            return make_response(
                jsonify(
                    {
                        "error": "Data given back by model was not valid JSON. Please try a different model."
                    }
                ),
                404,
            )
        # print(len(jsonResponses))
        if jsonResponses is not None:
            print(jsonResponses)
            for jsonResponse in jsonResponses:
                # print(jsonResponse)
                for errorNumber, errorPhrase in enumerate(jsonResponse.errorPhrases):
                    print(errorNumber, errorPhrase)
                    errorDescription = jsonResponse.errorExplanation[0]
                    reportInfo = reportInfo.replace(
                        errorPhrase, f"<mark> {errorPhrase} </mark>"
                    )
                    reportInfo = reportInfo.replace("\n", "</br>")
                    listOfErrors.append((errorPhrase, errorDescription))
            htmlResponse = render_template(
                "generatedResponse.html",
                correctedOutput=reportInfo,
                listOfErrors=listOfErrors,
            )
        else:
            htmlResponse = render_template(
                "generatedResponse.html",
                correctedOutput=f"<b>No errors were found in the report below.</b></br>{json.dumps(reportInfo)}",
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
