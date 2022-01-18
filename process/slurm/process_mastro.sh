#!/bin/bash
#SBATCH --job-name=PROC_OARS
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH -c 44
#SBATCH -n 1
#SBATCH -t 01:59:59
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics

source /cluster/home/jmarsill/.bashrc
source activate light7
file=PROCESS_"$(date +%Y_%m_%d_%T)".txt
dataset=quebec
path=/cluster/home/jmarsill/ProcessHN/process_hnradiomics.py
python $path --dataset $dataset > $file
