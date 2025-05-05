#!/bin/bash -l

#SBATCH -J analyzesae
#SBATCH -p gpu                
#SBATCH -t 00:20:00             
#SBATCH -N 1   
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=analyzesae.out

module load modules/2.3-20240529
module load gcc python/3.10.13
source ~/venvs/aion/bin/activate

srun python analyze_sae.py

