import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
import gym_microrts
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A2C agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsGlobalAgentMining24x24Prod-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=2000,
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                       help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=bool, default=False,
                       help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="gym-microrts",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--vf-coef', type=float, default=0.25,
                       help="value function's coefficient the loss function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                       help='the maximum norm for the gradient clipping')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help="policy entropy's coefficient the loss function")
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    wandb.save(os.path.abspath(__file__))

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)
# respect the default timelimit
if int(args.episode_length):
    if not isinstance(env, TimeLimit):
        env = TimeLimit(env, int(args.episode_length))
    else:
        env._max_episode_steps = int(args.episode_length)
else:
    args.episode_length = env._max_episode_steps if isinstance(env, TimeLimit) else 200
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(27, 16, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(32*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )

    def forward(self, x):
        x = torch.Tensor(np.moveaxis(x, -1, 1)).to(device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(27, 16, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.MaxPool2d(2),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(32*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = torch.Tensor(np.moveaxis(x, -1, 1)).to(device)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CategoricalMasked(Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        self.masks = torch.BoolTensor(masks).to(device)
        logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)
    

pg = Policy().to(device)
vf = Value().to(device)
optimizer = optim.Adam(list(pg.parameters()) + list(vf.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())
    actions = np.empty((args.episode_length,), dtype=object)
    rewards, dones = np.zeros((2, args.episode_length))
    obs = np.empty((args.episode_length,) + env.observation_space.shape)

    # ALGO LOGIC: put other storage logic here
    values = torch.zeros((args.episode_length), device=device)
    neglogprobs = torch.zeros((args.episode_length,), device=device)
    entropys = torch.zeros((args.episode_length,), device=device)

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        env.render()
        global_step += 1
        obs[step] = next_obs.copy()

        # ALGO LOGIC: put action logic here
        logits = pg.forward(obs[step:step+1])
        values[step] = vf.forward(obs[step:step+1])

        # ALGO LOGIC: `env.action_space` specific logic
        if isinstance(env.action_space, Discrete):
            probs = Categorical(logits=logits)
            action = probs.sample()
            actions[step], neglogprobs[step], entropys[step] = action.tolist()[0], -probs.log_prob(action), probs.entropy()

        elif isinstance(env.action_space, MultiDiscrete):
            logits_categories = torch.split(logits, env.action_space.nvec.tolist(), dim=1)
            action = []
            probs_categories = []
            probs_entropies = torch.zeros((logits.shape[0]), device=device)
            neglogprob = torch.zeros((logits.shape[0]), device=device)
            
            # Gym-microrts logic: unit selection masking
            i=0
            probs_categories.append(CategoricalMasked(logits=logits_categories[i], masks=env.unit_location_mask))
            if len(action) != env.action_space.shape:
                action.append(probs_categories[i].sample())
            neglogprob -= probs_categories[i].log_prob(action[i])
            probs_entropies += probs_categories[i].entropy()
            

            for i in range(1, len(logits_categories)):
                probs_categories.append(Categorical(logits=logits_categories[i]))
                if len(action) != env.action_space.shape:
                    action.append(probs_categories[i].sample())
                neglogprob -= probs_categories[i].log_prob(action[i])
                probs_entropies += probs_categories[i].entropy()
            action = torch.stack(action).transpose(0, 1).tolist()
            actions[step], neglogprobs[step], entropys[step] = action[0], neglogprob, probs_entropies

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards[step], dones[step], _ = env.step(actions[step])
        next_obs = np.array(next_obs)
        if dones[step]:
            break

    # ALGO LOGIC: training.
    # calculate the discounted rewards, or namely, returns
    returns = np.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0]-1)):
        returns[t] = rewards[t] + args.gamma * returns[t+1] * (1-dones[t])
    # advantages are returns - baseline, value estimates in our case
    advantages = returns - values.detach().cpu().numpy()

    vf_loss = loss_fn(torch.Tensor(returns).to(device), values) * args.vf_coef
    pg_loss = torch.Tensor(advantages).to(device) * neglogprobs
    loss = (pg_loss - entropys * args.ent_coef).mean() + vf_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(pg.parameters()) + list(vf.parameters()), args.max_grad_norm)
    optimizer.step()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/episode_reward", rewards.sum(), global_step)
    writer.add_scalar("losses/value_loss", vf_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropys[:step].mean().item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
# env.close()
writer.close()
