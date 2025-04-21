# Maintenance Manual

## Development

The technology stack used is Python based, and the environment used to manage packages was the Conda virtual environment. All development for this project was done using an M1 MacBook Air, apart from Jupyter Notebooks(file extension `.ipynb`), which were exported into Google Colab and ran. This is because they required more compute. 

### Conda environment

The Conda environment used is called `hons`. The `environment.yml` file acts as a single source of truth. 
To replicate the development environment:
1. Clone this repository.
2. `cd` into the `honours_project` directory
3. Run the command: 

```
conda env create --file environment.yml
```

> Note: To update the local environment after the `environment.yml` file has been updated, use the command ```conda env update --file environment.yml```. 

### Prerequisites

- A `.env` file that contains a Hugging Face token(`$HF_TOKEN`). You can create this with the commands `touch .env` and `nano .env` to edit the file if you are on a Unix machine or MacOS. An example `.env` file is shown in `.env.example`. The Hugging Face token can be acquired by going to the HuggingFace tokens page, located [here.](https://huggingface.co/settings/tokens)
- [Ollama](https://ollama.com/download) installed on your machine.
- [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install) - which can be run by following the instructions above.
- [TailwindCSS](https://tailwindcss.com/docs/installation/tailwind-cli) is a CSS library with pre-existing styles. It can be downloaded using `npm install tailwindcss @tailwindcss/cli`. After adding new styles to the `webview/templates` html files, you can use the command `npx @tailwindcss/cli -o webview/static/css/tailwindstyles.css` to update the .css file with the new styles. This is done so that there is always a local version of the CSS file inside the repository; allowing the report to work offline.  

### Tests

To run the builtin tests, run
```python3 -m unittest discover```. These tests, among other things, check whether the relevant files are present in the `./datasets` folder. 
 

### Directory Structure
#TODO
