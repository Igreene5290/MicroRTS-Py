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
  --wandb-project-name A2C-Tune2 \
  --num-selfplay-envs 256 \
  --num-bot-envs 4 \
  --num-steps 1024 \
  --learning-rate 5e-5 \
  --lr-schedule linear \
  --ent-coef 0.005 \
  --ent-coef-anneal True \
  --ent-coef-final 0.0005 \
  --gamma 0.995 \
  --gae-lambda 0.9 \
  --normalize-advantages True \
  --normalize-returns False \
  --acer-buffer-size 50000 \
  --acer-batch-size 4096 \
  --acer-rho-clip 1.0 \
  --reward-weight 10.0 1.0 1.0 0.2 1.0 1.0 \
  --total-timesteps 100000000 \
  --capture-video False

