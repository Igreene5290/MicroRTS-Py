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
  --cuda True \
  --prod-mode True \
  --wandb-project-name A2C-ACER-Explore \
  --num-selfplay-envs 64 \
  --num-bot-envs 4 \
  --num-steps 256 \
  --learning-rate 3e-4 \
  --ent-coef 0.05 \
  --gamma 0.999 \
  --gae-lambda 0.95 \
  --normalize-advantage True \
  --normalize-returns True \
  --acer-buffer-size 100000 \
  --acer-batch-size 8192 \
  --acer-rho-clip 10.0 \
  --reward-weight 100.0 1.0 1.0 0.2 1.0 10.0 \
  --total-timesteps 100000000 \
  --capture-video False
