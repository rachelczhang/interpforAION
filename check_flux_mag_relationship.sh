#!/bin/bash -l

#SBATCH -J checkfluxmag
#SBATCH -p gen                
#SBATCH -t 00:30:00             
#SBATCH -N 1   
#SBATCH --mem=20G                 
#SBATCH --output=checkfluxmag.out

module load modules/2.3-20240529
module load gcc python/3.10.13
source ~/venvs/aion/bin/activate

srun python check_flux_mag_relationship.py
