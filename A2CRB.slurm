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
  --wandb-project-name A2C-Tune1 \
  --num-selfplay-envs 128 \
  --num-bot-envs 4 \
  --num-steps 512 \
  --learning-rate 1e-4 \
  --ent-coef 0.01 \
  --gamma 0.995 \
  --gae-lambda 0.9 \
  --normalize-advantages True \
  --normalize-returns False \
  --acer-buffer-size 50000 \
  --acer-batch-size 4096 \
  --acer-rho-clip 3.0 \
  --reward-weight 10.0 1.0 1.0 0.2 1.0 1.0 \
  --total-timesteps 1e8 \
  --capture-video False

