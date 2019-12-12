#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=DwnLoad
#SBATCH --mail-type=END
##SBATCH --mail-user=lhm300@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/lhm300/
cd CV_project/sslime/extra_scripts

python3 create_stl10_data_files.py