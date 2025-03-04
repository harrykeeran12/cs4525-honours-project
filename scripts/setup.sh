#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=2 
#SBATCH --mem=8G
#SBATCH --ntasks-per-node=1 
#SBATCH --partition=gpu
#SBATCH --mail-user=u03he21@abdn.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=nlpRadioSetup
#SBATCH --output=environmentSetup.out

# This must be on the same partition as where the other files are going to be run.
module load conda/miniconda3

echo "Creating environment."

conda create -n hons python=3.12 -y
source activate base
conda activate hons
conda install -y -c conda-forge ollama-python python-dotenv huggingface_hub ollama pandas

