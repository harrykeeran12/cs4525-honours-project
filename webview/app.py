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


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/generate")
def generate():
    """Takes in the model name + report information as a form. Outputs the possible errors, as JSON."""
    SYSTEM = "You help correct radiology report errors. These include transcription errors, internal inconsistencies, insertion statements and translation errors. For every mistake found in the text, show the incorrect words and explain what the problem is. The errorPhrases array must be the same length as the errorType array and errorExplanation array. Ignore any errors deemed unnecessary."
    try:
        modelList = [model.get("model") for model in ollama.list().models]
    except Exception:
        return make_response(
            jsonify({"error": "Ollama not running. Please start ollama. "}), 400
        )

    modelName = request.form["modelName"]
    reportInfo = request.form["reportInformation"]
    if modelName in modelList:
        # return make_response(jsonify({"message": "Model name found."}), 200)
        # Ollama generates here:
        ollamaResponse = ollama.generate(
            model=modelName,
            prompt=SYSTEM + reportInfo,
            options={"temperature": 0},
            format=RadiologyError.model_json_schema(),
        )["response"]

        listOfErrors = []

        jsonResponse = json.loads(ollamaResponse)
        for errorNumber in range(len(jsonResponse["errorPhrases"])):
            errorPhrase = jsonResponse["errorPhrases"][errorNumber]
            errorDescription =  jsonResponse["errorExplanation"][errorNumber]
            print(errorPhrase)
            reportInfo = reportInfo.replace("\n", "<br></br>")
            reportInfo = reportInfo.replace(errorPhrase, f"<mark>{errorPhrase}</mark>")
            listOfErrors.append((errorPhrase, errorDescription))
        htmlResponse = f'<p class="px-2 py-3 h-[50vh] overflow-y-auto" id="correction">{reportInfo}</p>'

        htmlList = [f"<li class=\"flex flex-col gap-2\"><h2 class=\"font-semibold my-2\">⁉ {error[0]}</h2><p>{error[1]}</p></li>" for error in listOfErrors]

        unorderedList = f"<ul class=\"bg-blue-100 text-black-100 p-1 px-2 mt-5 rounded-lg\">{"<br></br>".join(htmlList)}</ul>"

        return make_response(
            htmlResponse+unorderedList,
            200,
        )
    else:
        return make_response(
            jsonify({"error": f"Model name {modelName} not present in ollama models."}),
            404,
        )


if __name__ == "__main__":
    app.run(debug=True)
