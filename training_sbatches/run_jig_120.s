#!/bin/bash
#BATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=09:00:00
#SBATCH --mem=50000
#SBATCH --job-name=J120
#SBATCH --mail-user=lhm300@nyu.edu
#SBATCH --mail-type=END
#SBATCH --output=jig/120/slurm_%j.out

OPT=$1 
#command line argument
. ~/.bashrc
module swap python3/intel  anaconda3/5.3.1
module load anaconda3/5.3.1
module load cudnn/9.0v7.0.5  
module load cuda/9.0.176 
conda activate slime_env
conda install -n slime_env nb_conda_kernels
module list
# conda activate 

cd /scratch/lhm300/CV_project/

export PYTHONPATH=sslime
python sslime/tools/train.py --config_file  sslime/extra_scripts/JigSaw/unsupervised_vgg_a_jigsaw_120_stl_10.yaml
python sslime/tools/train.py --config_file  sslime/extra_scripts/JigSaw/eval_vgg_a_jigsaw_120_stl_10.yaml