#!/bin/bash
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=2 
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1 
#SBATCH --partition=gpu 
#SBATCH --job-name=nlpRadio

module load conda/miniconda3

echo "Configuring ollama"
export PATH=${PATH}:${PWD}/ollama/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/ollama/lib
export OLLAMA_MODELS=${PWD}/ollama/models

echo $PWD

source activate base

conda activate hons

cd $HOME/sharedscratch
 
echo "Activating environment"

conda activate hons

echo "Running ollama background daemon:"

./bin/ollama serve &

sleep 10

./bin/ollama pull mistral:latest

./bin/ollama pull falcon3:latest

./bin/ollama pull qwen2.5:latest

srun python honours_project/prelim_eval.py 

conda deactivate

pkill ollama
exit 0
