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
  --wandb-project-name A2C-Tune4 \
  --num-selfplay-envs 128 \
  --num-bot-envs 4 \
  --num-steps 512 \
  --learning-rate 2e-6 \
  --ent-coef 0.001 \
  --gamma 0.995 \
  --gae-lambda 0.9 \
  --normalize-advantage True \
  --normalize-returns False \
  --acer-buffer-size 20000 \
  --acer-batch-size 2048 \
  --acer-rho-clip 0.5 \
  --reward-weight 10.0 1.0 1.0 0.2 1.0 1.0 \
  --total-timesteps 10000000 \
  --capture-video False

