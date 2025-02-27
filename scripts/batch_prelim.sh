#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=2 
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1 
#SBATCH --partition=gpu 
#SBATCH --job-name=nlpRadio
#SBATCH --output=prelimLog.out

module load conda/miniconda3

echo "Configuring ollama:"

# Configuring ollama variables.

echo $PWD
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_CONTEXT_LENGTH=1024
export PATH=${PATH}:${PWD}/ollama/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/ollama/lib
export OLLAMA_MODELS=${PWD}/ollama/models


source activate base

conda activate hons

cd $HOME/sharedscratch
 
echo "Activating environment"

conda activate hons

echo "Running ollama background daemon:"

./bin/ollama serve &

# Wait an amount of time so that the ollama server actually starts.
sleep 10

# Pull the models from the ollama servers.

./bin/ollama pull mistral:latest

./bin/ollama pull falcon3:latest

./bin/ollama pull qwen2.5:latest

# Use the slurm script to run the preliminary evaluation script. 

srun python honours_project/prelim_eval.py 

conda deactivate

pkill ollama
exit 0
