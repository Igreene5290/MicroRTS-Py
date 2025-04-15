#!/usr/bin/env python3
"""
A2C Implementation for MicroRTS using the same network backbone as PPO GridNet.
This version uses a single update per rollout (unlike PPO’s multiple epochs) and uses
the advantage actor–critic loss formulation.
Based on GridNet paper (Han et al. 2019) - http://proceedings.mlr.press/v97/han19a/han19a.pdf
"""

import argparse
import os
import random
import signal
import sys
import time
import subprocess
import wandb
from distutils.util import strtobool
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder, VecEnvWrapper
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# ---------------- Helper Classes and Functions ----------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='Experiment name')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
                        help='Gym environment id')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=1, help='Seed')
    parser.add_argument('--total-timesteps', type=int, default=50000000,
                        help='Total timesteps for training')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Use torch deterministic mode')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Enable CUDA if available')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='Production mode (e.g., wandb logging)')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='Capture video during training')
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts",
                        help="WandB project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="WandB entity/team")

    # Training hyperparameters:
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='Partial observability flag')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='Number of steps per rollout')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='Lambda for advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.02, help='Entropy coefficient')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum gradient norm')
    parser.add_argument('--normalize-advantage', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Normalize advantage')
    parser.add_argument('--normalize-returns', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Normalize returns')
    parser.add_argument('--lr-decay', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Linear decay of learning rate')
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Enable learning rate annealing')
    parser.add_argument('--num-models', type=int, default=100, help='Number of model checkpoints')
    # Add evaluation-related parameters:
    parser.add_argument('--max-eval-workers', type=int, default=4,
                        help='Maximum number of evaluation workers')
    parser.add_argument('--eval-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
                        help='Evaluation maps')
    parser.add_argument('--eval-frequency', type=int, default=1000000,
                        help='Frequency of evaluation in timesteps')
    parser.add_argument('--reward-weight', type=float, nargs='+',
                        default=[10.0, 1.0, 1.0, 0.2, 1.0, 4.0],
                        help='Reward weights')
    
    # Environment settings:
    parser.add_argument('--num-bot-envs', type=int, default=0,
                        help='Number of bot environments')
    parser.add_argument('--num-selfplay-envs', type=int, default=24,
                        help='Number of self-play environments')
    parser.add_argument('--train-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
                        help='Training maps')

    args = parser.parse_args()
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    args.save_frequency = max(1, int(args.num_updates // args.num_models))
    return args


# (These are copied nearly verbatim from your PPO implementation.)
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)

class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma
        try:
            base_env = env
            while hasattr(base_env, "venv"):
                base_env = base_env.venv
            self.rfs = getattr(base_env, 'reward_functions', ['WinLossRewardFunction'])
        except Exception:
            self.rfs = ['WinLossRewardFunction']

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos[:])
        for i in range(len(dones)):
            if "raw_rewards" in infos[i]:
                raw_rewards = infos[i]["raw_rewards"]
                self.raw_rewards[i] += [raw_rewards]
                self.raw_discount_rewards[i] += [
                    (self.gamma ** self.ts[i]) *
                    np.concatenate((raw_rewards, raw_rewards.sum()), axis=None)
                ]
            else:
                self.raw_rewards[i] += [np.array([float(rews[i])])]
                self.raw_discount_rewards[i] += [
                    (self.gamma ** self.ts[i]) * np.array([float(rews[i]), float(rews[i])])
                ]
            self.ts[i] += 1
            if dones[i]:
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs] if self.rfs is not None else []
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(0)
                raw_discount_names = (["discounted_" + str(rf) for rf in self.rfs] + ["discounted"]) if self.rfs is not None else []
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(dict(zip(raw_discount_names, raw_discount_returns)))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, mask_value=None):
        if masks is None:
            masks = torch.ones_like(logits, dtype=torch.bool)
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)

class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation
    def forward(self, x):
        return x.permute(self.permutation)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ---------------- A2C Network Definition ----------------
class Agent(nn.Module):
    def __init__(self, envs, mapsize=16 * 16):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        h, w, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(inplace=False),
            layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 128)),
            nn.ReLU(inplace=False),
            layer_init(nn.Linear(128, 1), std=1),
        )
        self.register_buffer("mask_value", torch.tensor(-1e8))
    
    def get_action_and_value(self, x, action=None, invalid_action_masks=None, envs=None, device=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.reshape(-1, envs.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envs.action_plane_space.nvec.tolist(), dim=1)
        if action is None:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)
            multi_categoricals = [
                CategoricalMasked(logits=logit, masks=mask, mask_value=self.mask_value)
                for logit, mask in zip(split_logits, split_invalid_action_masks)
            ]
            action = torch.stack([cat.sample() for cat in multi_categoricals])
        else:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(invalid_action_masks, envs.action_plane_space.nvec.tolist(), dim=1)
            multi_categoricals = [
                CategoricalMasked(logits=logit, masks=mask, mask_value=self.mask_value)
                for logit, mask in zip(split_logits, split_invalid_action_masks)
            ]
        logprob = torch.stack([cat.log_prob(a) for a, cat in zip(action, multi_categoricals)])
        entropy = torch.stack([cat.entropy() for cat in multi_categoricals])
        num_predicted_parameters = len(envs.action_plane_space.nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)
        value = self.critic(hidden)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks, value

    def get_value(self, x):
        return self.critic(self.encoder(x))

# ---------------- Evaluation Helper ----------------
def run_evaluation(model_path: str, output_path: str, eval_maps: List[str]):
    args = [
        "python",
        "league.py",
        "--evals",
        model_path,
        "--update-db",
        "false",
        "--cuda",
        "false",
        "--output-path",
        output_path,
        "--model-type",
        "a2c",  # Ensure we pass the appropriate model type
        "--maps",
        *eval_maps,
    ]
    fd = subprocess.Popen(args)
    print(f"Evaluating {model_path}")
    return_code = fd.wait()
    assert return_code == 0
    return (model_path, output_path)

class TrueskillWriter:
    def __init__(self, prod_mode, writer, league_path: str, league_step_path: str):
        self.prod_mode = prod_mode
        self.writer = writer
        self.trueskill_df = pd.read_csv(league_path)
        self.trueskill_step_df = pd.read_csv(league_step_path)
        self.trueskill_step_df["type"] = self.trueskill_step_df["name"]
        self.trueskill_step_df["step"] = 0
        # xxx(okachaiev): not sure we need this copy
        self.preset_trueskill_step_df = self.trueskill_step_df.copy()

    def on_evaluation_done(self, future):
        if future.cancelled():
            return
        model_path, output_path = future.result()
        league = pd.read_csv(output_path, index_col="name")
        assert model_path in league.index
        model_global_step = int(model_path.split("/")[-1][:-3])
        self.writer.add_scalar("charts/trueskill", league.loc[model_path]["trueskill"], model_global_step)
        print(f"global_step={model_global_step}, trueskill={league.loc[model_path]['trueskill']}")

        # table visualization logic
        if self.prod_mode:
            trueskill_data = {
                "name": league.loc[model_path].name,
                "mu": league.loc[model_path]["mu"],
                "sigma": league.loc[model_path]["sigma"],
                "trueskill": league.loc[model_path]["trueskill"],
            }
            self.trueskill_df = self.trueskill_df.append(trueskill_data, ignore_index=True)
            wandb.log({"trueskill": wandb.Table(dataframe=self.trueskill_df)})
            trueskill_data["type"] = "training"
            trueskill_data["step"] = model_global_step
            self.trueskill_step_df = self.trueskill_step_df.append(trueskill_data, ignore_index=True)
            preset_trueskill_step_df_clone = self.preset_trueskill_step_df.copy()
            preset_trueskill_step_df_clone["step"] = model_global_step
            self.trueskill_step_df = self.trueskill_step_df.append(preset_trueskill_step_df_clone, ignore_index=True)
            wandb.log({"trueskill_step": wandb.Table(dataframe=self.trueskill_step_df)})

# ---------------- Signal Handler to Save on Interrupt ----------------
def save_model(agent, experiment_name, global_step):
    model_path = f"models/{experiment_name}/agent.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved at {model_path} (global step: {global_step})")

def signal_handler(sig, frame, agent, experiment_name, global_step):
    save_model(agent, experiment_name, global_step)
    sys.exit(0)

# --------------------- A2C Training Loop ---------------------
def train():
    args = parse_args()
    print(f"Save frequency: {args.save_frequency}")

    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    # Set up signal handlers (Ctrl+C, kill signals)
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, agent, experiment_name, global_step))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, agent, experiment_name, global_step))

    # Set seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Set up environment.
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=5000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 2)]
             + [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 1))]
             + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 1))],
        map_paths=args.train_maps,
        reward_weight=np.array(args.reward_weight),
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(envs, f"videos/{experiment_name}",
                                record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)
    assert isinstance(envs.action_space, MultiDiscrete), "Only MultiDiscrete action space is supported"

    # PATCH: Add a dummy source_unit_mask to the base env if absent.
    base_env = envs
    while hasattr(base_env, "venv"):
        base_env = base_env.venv
    nenvs = getattr(base_env, "num_envs", 1)
    if isinstance(base_env.action_space, MultiDiscrete):
        total_subactions = int(np.sum(base_env.action_space.nvec))
        dummy_mask = np.ones((nenvs, total_subactions + 1), dtype=np.int32)
        base_env.source_unit_mask = dummy_mask
        envs.source_unit_mask = dummy_mask
    else:
        base_env.source_unit_mask = np.ones(nenvs, dtype=np.int32)
        envs.source_unit_mask = np.ones(nenvs, dtype=np.int32)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Set up learning rate decay if enabled.
    if args.lr_decay:
        def lr_lambda(step):
            return max(1.0 - float(step)/args.num_updates, 0)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Set up evaluation executor.
    eval_executor = None
    if args.max_eval_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        eval_executor = ThreadPoolExecutor(max_workers=args.max_eval_workers, thread_name_prefix="league-eval-")

    # Derived training variables.
    num_envs = args.num_selfplay_envs + args.num_bot_envs
    batch_size = num_envs * args.num_steps


    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)

    trueskill_writer = TrueskillWriter(
        args.prod_mode, writer, "gym-microrts-static-files/league.csv", "gym-microrts-static-files/league.csv"
    )

    # Main training loop.
    for update in range(1, args.num_updates + 1):
        # Adjust learning rate if annealing is enabled.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / args.num_updates
            lr_now = args.learning_rate * frac
            optimizer.param_groups[0]["lr"] = lr_now
        if args.lr_decay:
            scheduler.step()

        # Rollout collection.
        obs_list = []
        actions_list = []
        logprobs_list = []
        values_list = []
        rewards_list = []
        dones_list = []
        for step in range(args.num_steps):
            obs_list.append(next_obs)
            invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).to(device)
            action, logp, entropy, inv_masks, value = agent.get_action_and_value(
                next_obs, invalid_action_masks=invalid_action_masks, envs=envs, device=device
            )
            actions_list.append(action)
            logprobs_list.append(logp)
            values_list.append(value)
            action_np = action.cpu().numpy()
            next_obs_np, reward, done, infos = envs.step(action_np.reshape(num_envs, -1))
            rewards_list.append(torch.Tensor(reward).to(device))
            dones_list.append(torch.Tensor(done).to(device))
            global_step += num_envs
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)
            for info in infos:
                if "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # Compute bootstrap value.
        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze()
        # Stack rollout.
        rollout_obs = torch.stack(obs_list)                # (num_steps, num_envs, H, W, C)
        rollout_logprobs = torch.stack(logprobs_list)        # (num_steps, num_envs)
        rollout_values = torch.stack(values_list).squeeze()  # (num_steps, num_envs)
        rollout_rewards = torch.stack(rewards_list)          # (num_steps, num_envs)
        rollout_dones = torch.stack(dones_list)              # (num_steps, num_envs)

        # Compute advantages using GAE.
        returns = torch.zeros_like(rollout_rewards)
        advantages = torch.zeros_like(rollout_rewards)
        gae = torch.zeros(num_envs, device=device)
        for t in reversed(range(args.num_steps)):
            next_val = next_value if t == args.num_steps - 1 else rollout_values[t+1]
            delta = rollout_rewards[t] + args.gamma * next_val * (1 - rollout_dones[t]) - rollout_values[t]
            gae = delta + args.gamma * args.gae_lambda * (1 - rollout_dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + rollout_values[t]

        # Optionally normalize advantages.
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Flatten rollout.
        flat_logprobs = rollout_logprobs.view(-1)
        flat_values = rollout_values.view(-1)
        flat_returns = returns.view(-1)
        flat_advantages = advantages.view(-1)

        policy_loss = -(flat_logprobs * flat_advantages.detach()).mean()
        value_loss = ((flat_returns - flat_values) ** 2).mean()
        entropy_loss = - rollout_logprobs.mean()
        total_loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        writer.add_scalar("loss/policy", policy_loss.item(), global_step)
        writer.add_scalar("loss/value", value_loss.item(), global_step)
        writer.add_scalar("loss/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("loss/total", total_loss.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
        print(f"Update {update}: Global Steps: {global_step}, Total Loss: {total_loss.item():.4f}")

        # -----------------------------------------------------

        if (update - 1) % args.save_frequency == 0:
            model_path = f"models/{experiment_name}/{global_step}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(agent.state_dict(), f"models/{experiment_name}/agent.pt")
            torch.save(agent.state_dict(), f"models/{experiment_name}/{global_step}.pt")
            print(f"Model checkpoint saved at {model_path}")
            if eval_executor is not None:
                future = eval_executor.submit(
                    run_evaluation,
                    f"models/{experiment_name}/{global_step}.pt",
                    f"runs/{experiment_name}/{global_step}.csv",
                    args.eval_maps,
                )
                print(f"Queued models/{experiment_name}/{global_step}.pt")
                future.add_done_callback(trueskill_writer.on_evaluation_done)

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per update: {total_time / args.num_updates:.2f} seconds")
    
    final_model_path = f"models/{experiment_name}/agent.pt"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(agent.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    writer.close()
    envs.close()
    if eval_executor is not None:
        eval_executor.shutdown(wait=True, cancel_futures=False)

if __name__ == "__main__":
    train()
