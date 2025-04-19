#!/bin/sh
#SBATCH --gpus-per-node=1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

cd ~/MicroRTS-Py

source .venv/bin/activate
export JAVA_HOME=/usr/lib/jvm/jre/

uv pip install -r requirements.txt
pwd > .venv/lib/python3.9/site-packages/gym_microrts.pth
python hello_world.py

cd experiments
python a2c2.py \
  --num-selfplay-envs 4 \         
  --num-steps 128 \                
  --acer-buffer-size 2000 \        
  --acer-batch-size 128 \         
  --total-timesteps 10000000 \
  --cuda True
