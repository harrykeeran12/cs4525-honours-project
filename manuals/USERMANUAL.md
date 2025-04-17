# User Manual

## Prerequisites:
The radiology tool requires for Ollama and a Flask server to be installed. 


1. Click the link [here](https://ollama.com/download) to the Ollama homepage.
2. Click the download button for the specific operating system.
3. Run through the setup window. 
4. Run the Ollama background server(daemon). 


## Running the radiology tool.

1. Run the Ollama server using `ollama serve &`. This allows you to run the Ollama server in the background. 
2. Rename the `.env.example` file to `.env`. This is because the Flask server is inside the `webview` folder.
3. Activate the conda environment `conda activate hons`. 
4. 