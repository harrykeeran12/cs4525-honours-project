#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=2 
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1 
#SBATCH --partition=gpu
#SBATCH --job-name=nlpRadioSetup
#SBATCH --output=environmentSetup.out

# This must be on the same partition as where the other files are going to be run.
module load conda/miniconda3

echo "Creating environment."

conda create -n hons python=3.12
conda activate hons
conda install -c conda-forge ollama-python python-dotenv huggingface_hub ollama pandas

