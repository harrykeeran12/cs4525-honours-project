from pydantic import ValidationError
from ..schema import RadiologyErrors
from ..prompts import SYSTEM, SYSTEM2, SYSTEM3
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
