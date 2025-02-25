#!/bin/bash

module load conda/miniconda3

echo "Creating environment."

conda env create --file honours_project/environment.yml

