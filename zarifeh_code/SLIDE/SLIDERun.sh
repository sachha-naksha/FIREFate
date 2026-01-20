#!/bin/bash
#SBATCH -t 12-00:00

#SBATCH --job-name=SLIDE
# partition (queue) declaration
#SBATCH --mail-user=''
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=256g

module load gcc/10.2.0
module load r/4.3.0



Rscript .../SLIDE/SLIDERun.R

