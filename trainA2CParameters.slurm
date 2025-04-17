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
  --total-timesteps 50000000 \
  --learning-rate 2.5e-4 \
  --num-steps 512 \
  --gamma 0.99 \
  --ent-coef 0.1 \
  --reward-weight 10.0 1.0 1.0 2.0 1.0 4.0 \
  --num-selfplay-envs 32 \
  --cuda True
