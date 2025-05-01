#!/bin/bash -l

#SBATCH -J saveact
#SBATCH -p gpu                
#SBATCH -t 00:30:00             
#SBATCH -N 1   
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=saveact.out

module load modules/2.3-20240529
module load gcc python/3.10.13
source ~/venvs/aion/bin/activate

srun python save_activations.py
