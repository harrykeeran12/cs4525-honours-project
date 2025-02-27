# Honours Project

This project aims to use smaller LLMs to correct and find errors in radiology reports.

The dataset used is located [here.](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/tree/main/dataset/radiology_text_reports)



## Development

### Prerequisites:

- A `.env` file that contains a Hugging Face token(`$HF_TOKEN`). You can create this with the commands `touch .env` and `nano .env` to edit the file if you are on a Unix machine or MacOS. An example `.env` file is shown in `.env.example`. 
- [Ollama](https://ollama.com/download) installed on your machine.
- Conda to replicate the development environment on your machine. To replicate the environment clone the repository, `cd` into the directory, and run the command: 

```
conda env create --file environment.yml
```
or

```
conda create -n hons python=3.12
conda activate hons
conda install -c conda-forge ollama-python python-dotenv huggingface_hub ollama pandas
```

The preliminary evaluation results were run on a HPC supercomputer. As such, the scripts located in the `scripts/` directory were designed to be run on a Linux machine. 

## Running Locally

`TODO`
