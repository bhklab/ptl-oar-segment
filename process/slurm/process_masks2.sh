#!/bin/bash
#SBATCH --job-name=PROC_OARS
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH -c 44
#SBATCH -n 1
#SBATCH -t 02:59:59
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics

source /cluster/home/jmarsill/.bashrc
source activate light8
file=PROCESS_"$(date +%Y_%m_%d_%T)".txt
path=/cluster/home/jmarsill/ProcessHN/process_masks2.py
vol=OARS
site=ALL
data=radcure
output=/cluster/projects/radiomics/Temp/OAR0820/masks/
input=/cluster/projects/radiomics/RADCURE-images/

python $path --volume $vol --site $site --dataset $data --output $output --input $input > $file
