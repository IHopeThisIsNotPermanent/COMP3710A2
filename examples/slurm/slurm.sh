#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=Part3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

source /home/Student/s4641797/conda/bin/activate /home/Student/s4641797/python_envs/a2
python Part3.py


