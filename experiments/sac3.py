#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) implementation for MicroRTS, using the same network sizes as the PPO GridNet.
Based on the GridNet paper (Han et al. 2019): http://proceedings.mlr.press/v97/han19a/han19a.pdf
"""

import argparse
import os
import random
import time
import uuid
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder, VecEnvWrapper
from gym.spaces import MultiDiscrete

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

# --------------------- Argument Parsing ---------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment and system parameters.
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='Name of the experiment')
    parser.add_argument('--gym-id', type=str, default="MicroRTSGridModeVecEnv",
                        help='Gym environment id (display only)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--total-timesteps', type=int, default=50000000,
                        help='Total timesteps for training')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Set torch.backends.cudnn.deterministic=True')
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
    # SAC hyperparameters.
    parser.add_argument('--reward-weight', type=float, nargs='+', default=[10.0, 1.0, 1.0, 0.2, 1.0, 4.0],
                        help='Reward weights')
    parser.add_argument('--partial-obs', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='Partial observability flag')
    parser.add_argument('--num-bot-envs', type=int, default=0,
                        help='Number of bot environments')
    parser.add_argument('--num-selfplay-envs', type=int, default=24,
                        help='Number of self-play environments')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='Number of steps per environment rollout')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target network update rate')
    parser.add_argument('--buffer-size', type=int, default=50000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for SAC updates')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Initial entropy coefficient')
    parser.add_argument('--auto-alpha', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Enable automatic entropy tuning')
    parser.add_argument('--target-entropy-scale', type=float, default=0.98,
                        help='Target entropy scale for auto alpha')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='Reward scale factor')
    parser.add_argument('--num-models', type=int, default=100,
                        help='Number of checkpoints to save')
    parser.add_argument('--max-eval-workers', type=int, default=4,
                        help='Maximum number of evaluation workers')
    parser.add_argument('--train-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
                        help='Training maps')
    parser.add_argument('--eval-maps', nargs='+', default=["maps/16x16/basesWorkers16x16A.xml"],
                        help='Evaluation maps')
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='Enable learning rate annealing')
    args = parser.parse_args()
    # Derived parameters.
    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    args.save_frequency = max(1, int(args.total_timesteps // (args.num_models * args.num_steps * args.num_envs)))
    return args

# --------------------- Environment Statistics Recorder ---------------------
class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.rfs = env.rfs if hasattr(env, 'rfs') else None

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
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i]) *
                np.concatenate((infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()), axis=None)
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

# --------------------- SAC Helper Classes and Functions ---------------------
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None, mask_value=None):
        if masks is not None:
            if logits is not None:
                logits = torch.where(masks.bool(), logits, mask_value)
            else:
                probs = torch.where(masks.bool(), probs, torch.zeros_like(probs))
                probs = probs / probs.sum(dim=-1, keepdim=True)
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

# --------------------- SAC Network Definitions ---------------------
class SACEncoder(nn.Module):
    def __init__(self, observation_shape):
        super().__init__()
        h, w, c = observation_shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=False),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
    def forward(self, x):
        return self.encoder(x)

class SoftQNetwork(nn.Module):
    def __init__(self, encoder, action_shape):
        super().__init__()
        self.encoder = encoder
        self.flatten = nn.Flatten()
        self.q = nn.Sequential(
            layer_init(nn.Linear(64 * 4 * 4 + int(np.prod(action_shape)), 256)),
            nn.ReLU(inplace=False),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(inplace=False),
            layer_init(nn.Linear(256, 1), std=1),
        )
    def forward(self, x, a):
        x = self.encoder(x)
        x = self.flatten(x)
        a_flat = a.reshape(a.shape[0], -1)
        x = torch.cat([x, a_flat], dim=1)
        return self.q(x)

class Actor(nn.Module):
    def __init__(self, encoder, action_plane_space, mapsize=16*16):
        super().__init__()
        self.encoder = encoder
        self.mapsize = mapsize
        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(inplace=False),
            layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )
        self.action_plane_space = action_plane_space
        self.register_buffer("mask_value", torch.tensor(-1e8))
    def forward(self, x, invalid_action_masks=None, deterministic=False, action=None):
        hidden = self.encoder(x)
        logits = self.actor(hidden)
        # Reshape to (batch * mapsize, total_action_dim)
        grid_logits = logits.reshape(-1, self.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, self.action_plane_space.nvec.tolist(), dim=1)
        if invalid_action_masks is not None:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks, self.action_plane_space.nvec.tolist(), dim=1)
            multi_categoricals = [
                CategoricalMasked(logits=logit, masks=mask, mask_value=self.mask_value)
                for logit, mask in zip(split_logits, split_invalid_action_masks)
            ]
        else:
            multi_categoricals = [Categorical(logits=logit) for logit in split_logits]
        if action is None:
            if deterministic:
                action = torch.stack([torch.argmax(cat.probs, dim=1) for cat in multi_categoricals])
            else:
                action = torch.stack([cat.sample() for cat in multi_categoricals])
        logprob = torch.stack([cat.log_prob(a) for a, cat in zip(action, multi_categoricals)])
        entropy = torch.stack([cat.entropy() for cat in multi_categoricals])
        num_predicted_parameters = len(self.action_plane_space.nvec)
        logprob = logprob.T.view(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.view(-1, self.mapsize, num_predicted_parameters)
        action = action.T.view(-1, self.mapsize, num_predicted_parameters)
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1)

class Agent(nn.Module):
    def __init__(self, envs, device=None, mapsize=16*16):
        super().__init__()
        self.mapsize = mapsize
        self.device = device if device is not None else torch.device("cpu")
        # Shared encoder for actor and critics
        self.encoder = SACEncoder(envs.observation_space.shape).to(self.device)
        # Actor network
        self.actor = Actor(self.encoder, envs.action_plane_space, mapsize).to(self.device)
        # Twin Q-networks
        self.qf1 = SoftQNetwork(self.encoder, (mapsize, len(envs.action_plane_space.nvec))).to(self.device)
        self.qf2 = SoftQNetwork(self.encoder, (mapsize, len(envs.action_plane_space.nvec))).to(self.device)
        self.qf1_target = SoftQNetwork(SACEncoder(envs.observation_space.shape).to(self.device),
                                       (mapsize, len(envs.action_plane_space.nvec))).to(self.device)
        self.qf2_target = SoftQNetwork(SACEncoder(envs.observation_space.shape).to(self.device),
                                       (mapsize, len(envs.action_plane_space.nvec))).to(self.device)
        # Copy parameters to target networks
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        # Freeze target networks
        for param in self.qf1_target.parameters():
            param.requires_grad = False
        for param in self.qf2_target.parameters():
            param.requires_grad = False

    def get_action(self, x, invalid_action_masks=None, deterministic=False):
        with torch.no_grad():
            action, _, _ = self.actor(x, invalid_action_masks, deterministic)
        return action

    def get_action_and_value(self, x, invalid_action_masks=None, deterministic=False, action=None):
        action, log_prob, entropy = self.actor(x, invalid_action_masks, deterministic, action)
        q1_value = self.qf1(x, action)
        q2_value = self.qf2(x, action)
        q_value = torch.min(q1_value, q2_value)
        return action, log_prob, entropy, q_value, q1_value, q2_value

# --------------------- Replay Buffer ---------------------
class ReplayBuffer:
    def __init__(self, buffer_size, observation_shape, action_shape, device, mask_size):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.observations = np.zeros((buffer_size,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size,) + observation_shape, dtype=np.float32)
        self.invalid_action_masks = np.zeros((buffer_size, mask_size), dtype=np.float32)
        self.next_invalid_action_masks = np.zeros((buffer_size, mask_size), dtype=np.float32)
        self.device = device

    def add(self, obs, action, reward, next_obs, done, invalid_action_mask, next_invalid_action_mask):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.invalid_action_masks[self.ptr] = invalid_action_mask
        self.next_invalid_action_masks[self.ptr] = next_invalid_action_mask
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer. Current size: {self.size}, requested batch size: {batch_size}")
        
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.observations[idxs]).to(self.device),
            torch.FloatTensor(self.actions[idxs]).to(self.device),
            torch.FloatTensor(self.rewards[idxs]).to(self.device),
            torch.FloatTensor(self.next_observations[idxs]).to(self.device),
            torch.FloatTensor(self.dones[idxs]).to(self.device),
            torch.FloatTensor(self.invalid_action_masks[idxs]).to(self.device),
            torch.FloatTensor(self.next_invalid_action_masks[idxs]).to(self.device)
        )

# --------------------- SAC Training Loop ---------------------
def run_training():
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}_{uuid.uuid4().hex[:8]}"
    if args.prod_mode:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", 
                    "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]))
    
    # Set seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set up environment.
    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_selfplay_envs,
        num_bot_envs=args.num_bot_envs,
        partial_obs=args.partial_obs,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
             + [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 2))]
             + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))]
             + [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
        map_paths=args.train_maps,
        reward_weight=np.array(args.reward_weight),
        cycle_maps=args.train_maps,
    )
    envs = MicroRTSStatsRecorder(envs, args.gamma)
    envs = VecMonitor(envs)
    if args.capture_video:
        envs = VecVideoRecorder(envs, f"videos/{run_name}", 
                                record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)
    
    # Define shapes.
    mapsize = 16 * 16
    # For discrete actions: use len(nvec); for invalid masks use sum(nvec)
    action_shape = (mapsize, len(envs.action_plane_space.nvec))
    mask_size = mapsize * int(np.sum(envs.action_plane_space.nvec))
    
    # Initialize replay buffer with smaller initial size
    initial_buffer_size = min(args.buffer_size, 10000)  # Start with a smaller buffer
    rb = ReplayBuffer(initial_buffer_size, envs.observation_space.shape, action_shape, device, mask_size)
    
    # Instantiate agent
    agent = Agent(envs, device, mapsize)
    
    # Set up optimizers with gradient clipping
    critic_params = list(agent.qf1.parameters()) + list(agent.qf2.parameters())
    critic_optimizer = optim.Adam(critic_params, lr=args.learning_rate)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
    
    # Enable gradient clipping
    max_grad_norm = 1.0
    
    # Automatic entropy tuning
    if args.auto_alpha:
        target_entropy = -np.prod(action_shape) * args.target_entropy_scale
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
    else:
        alpha = args.alpha
    
    # (Optional) Evaluation executor.
    eval_executor = None
    if args.max_eval_workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        eval_executor = ThreadPoolExecutor(max_workers=args.max_eval_workers, thread_name_prefix="league-eval-")
    
    # Number of updates.
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    
    print("Model's state_dict:")
    for param_tensor in agent.state_dict():
        print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
    total_params = sum([param.nelement() for param in agent.parameters()])
    print("Model's total parameters:", total_params)
    
    obs = envs.reset()
    global_step = 0
    start_time = time.time()
    
    # Main training loop.
    for update in range(1, num_updates + 1):
        # Optionally anneal learning rate.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            for opt in [critic_optimizer, actor_optimizer]:
                opt.param_groups[0]["lr"] = lr_now
            if args.auto_alpha:
                alpha_optimizer.param_groups[0]["lr"] = lr_now
        
        # Experience collection.
        for step in range(args.num_steps):
            global_step += args.num_envs
            invalid_action_masks = torch.tensor(np.array(envs.get_action_mask())).to(device)
            obs_tensor = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                action, log_prob, entropy, q_value, _, _ = agent.get_action_and_value(obs_tensor, invalid_action_masks)
            actions_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = envs.step(actions_np.reshape(envs.num_envs, -1))
            # Add transitions to replay buffer.
            for i in range(args.num_envs):
                rb.add(
                    obs[i],
                    actions_np[i],
                    rewards[i] * args.reward_scale,
                    next_obs[i],
                    dones[i],
                    invalid_action_masks[i].cpu().numpy().flatten(),
                    np.array(envs.get_action_mask()[i]).flatten()
                )
            obs = next_obs
            # Log episodic returns.
            for info in infos:
                if "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break
        
        # SAC update loop.
        if rb.size > args.batch_size:
            cum_qf1_loss = 0
            cum_qf2_loss = 0
            cum_actor_loss = 0
            cum_alpha_loss = 0
            for _ in range(args.num_steps):
                try:
                    observations, actions, rewards, next_observations, dones, iam, next_iam = rb.sample(args.batch_size)
                    # Reshape masks to (batch_size, mapsize, -1)
                    mask_shape = (args.batch_size, mapsize, int(np.sum(envs.action_plane_space.nvec)))
                    iam = iam.reshape(mask_shape)
                    next_iam = next_iam.reshape(mask_shape)
                    
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _, _, _, _ = agent.get_action_and_value(
                            next_observations, next_iam, deterministic=False
                        )
                        next_q1 = agent.qf1_target(next_observations, next_state_actions)
                        next_q2 = agent.qf2_target(next_observations, next_state_actions)
                        next_q = torch.min(next_q1, next_q2) - alpha * next_state_log_pi
                        target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * args.gamma * next_q
                    
                    current_q1 = agent.qf1(observations, actions)
                    current_q2 = agent.qf2(observations, actions)
                    qf1_loss = ((current_q1 - target_q) ** 2).mean()
                    qf2_loss = ((current_q2 - target_q) ** 2).mean()
                    critic_loss = qf1_loss + qf2_loss

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
                    critic_optimizer.step()
                    
                    actor_actions, log_pi, _, _, _, _ = agent.get_action_and_value(observations, iam, deterministic=False)
                    q1_pi = agent.qf1(observations, actor_actions)
                    q2_pi = agent.qf2(observations, actor_actions)
                    min_q_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha * log_pi - min_q_pi).mean()
                    
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_grad_norm)
                    actor_optimizer.step()
                    
                    if args.auto_alpha:
                        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
                        cum_alpha_loss += alpha_loss.item()
                    
                    cum_qf1_loss += qf1_loss.item()
                    cum_qf2_loss += qf2_loss.item()
                    cum_actor_loss += actor_loss.item()
                    
                    # Soft update target networks
                    for param, target_param in zip(agent.qf1.parameters(), agent.qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(agent.qf2.parameters(), agent.qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                except Exception as e:
                    print(f"Error during update: {e}")
                    continue
            
            avg_qf1_loss = cum_qf1_loss / args.num_steps
            avg_qf2_loss = cum_qf2_loss / args.num_steps
            avg_actor_loss = cum_actor_loss / args.num_steps
            critic_loss_avg = avg_qf1_loss + avg_qf2_loss
            writer.add_scalar("loss/qf1", avg_qf1_loss, global_step)
            writer.add_scalar("loss/qf2", avg_qf2_loss, global_step)
            writer.add_scalar("loss/critic", critic_loss_avg, global_step)
            writer.add_scalar("loss/actor", avg_actor_loss, global_step)
            if args.auto_alpha:
                avg_alpha_loss = cum_alpha_loss / args.num_steps
                writer.add_scalar("loss/alpha", avg_alpha_loss, global_step)
                writer.add_scalar("charts/alpha", alpha, global_step)
            writer.add_scalar("charts/learning_rate", actor_optimizer.param_groups[0]["lr"], global_step)
            print(f"Update {update}: Global Steps = {global_step}, Critic Loss = {critic_loss_avg:.4f}, Actor Loss = {avg_actor_loss:.4f}")
        
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        
        # Save model checkpoint periodically.
        if update % args.save_frequency == 0:
            checkpoint_path = f"models/{run_name}/{global_step}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'qf1_optimizer_state_dict': critic_optimizer.state_dict(),
                'qf2_optimizer_state_dict': actor_optimizer.state_dict(),
                'actor_optimizer_state_dict': alpha_optimizer.state_dict() if args.auto_alpha else None,
                'log_alpha': log_alpha if args.auto_alpha else None,
                'args': vars(args),
                'update': update,
                'global_step': global_step,
            }, checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per update: {total_time/num_updates:.2f} seconds")
    
    final_path = f"models/{run_name}/final.pt"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'qf1_optimizer_state_dict': critic_optimizer.state_dict(),
        'qf2_optimizer_state_dict': actor_optimizer.state_dict(),
        'actor_optimizer_state_dict': alpha_optimizer.state_dict() if args.auto_alpha else None,
        'log_alpha': log_alpha if args.auto_alpha else None,
        'args': vars(args),
        'update': update,
        'global_step': global_step,
    }, final_path)
    print(f"Final model saved to {final_path}")
    
    writer.close()
    envs.close()
    if eval_executor is not None:
        eval_executor.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    run_training()
