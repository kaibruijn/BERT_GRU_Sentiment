#!/bin/bash
#SBATCH --time=16:59:55
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=15G
#SBATCH --mail-type=ALL

# Run stuff here
module load Python/3.6.4-foss-2018a
python bert_gru.py
