#!/bin/bash -l

#SBATCH -J trainsae
#SBATCH -p gpu                
#SBATCH -t 10:00:00             
#SBATCH -N 1   
#SBATCH --gpus=1
#SBATCH --mem=50G                 
#SBATCH --output=trainsae.out

module load modules/2.3-20240529
module load gcc python/3.10.13
source ~/venvs/aion/bin/activate

srun python train_sae.py
