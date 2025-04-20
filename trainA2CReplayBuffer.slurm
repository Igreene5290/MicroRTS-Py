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
  --prod-mode True \
  --wandb-project-name microrts-acer-test \
  --num-selfplay-envs 32 \
  --num-steps 256 \
  --learning-rate 7e-4 \
  --ent-coef 0.01 \
  --acer-buffer-size 20000 \
  --acer-batch-size 512 \
  --acer-rho-clip 10.0
