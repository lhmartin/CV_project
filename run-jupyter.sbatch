#!/bin/bash

#SBATCH --job-name=jupyterTest2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p40:1



module purge
#module load jupyter-kernels/py2.7
module load jupyter-kernels/py3.5
module load torchvision/python3.5/0.1.9

. ~/.bashrc
module swap python3/intel  anaconda3/5.3.1
module load anaconda3/5.3.1
module load cudnn/9.0v7.0.5
module load cuda/9.0.176
conda activate slime_env
conda install -n slime_env nb_conda_kernels

module list


port=$(shuf -i 6000-9999 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

cat<<EOF
Jupyter server is running on: $(hostname)
Job starts at: $(date)
Step 1 :
If you are working in NYU campus, please open an iTerm window, run command
ssh -L $port:localhost:$port $USER@prince.hpc.nyu.edu
If you are working off campus, you should already have ssh tunneling setup through HPC bastion host, 
that you can directly login to prince with command
ssh $USER@prince
Please open an iTerm window, run command
ssh -L $port:localhost:$port $USER@prince
Step 2:
Keep the iTerm windows in the previouse step open. Now open browser, find the line with
The Jupyter Notebook is running at: $(hostname)
the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)
you should be able to connect to jupyter notebook running remotly on prince compute node with above url
EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)


