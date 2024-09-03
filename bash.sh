#!/bin/bash

# Request an interactive session with specific parameters
#SBATCH --partition=a100
#SBATCH --nodelist=mp-gpu4-a100-2
#SBATCH --job-name=alpha_rec


# Run the python script with a time limit
python run.py > out.out