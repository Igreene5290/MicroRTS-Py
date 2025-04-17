# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import argparse
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
from classes.DataCollector import DecisionTransformerGymDataCollator
from datasets import DatasetDict
from dt_gridnet_eval import decode_action, decode_obs, get_action
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from experiments.classes.TrainableDT import TrainableDT
from gym_microrts import microrts_ai  # noqa


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--dt-dataset', type=str, default="episode_data/cm-mcrp-dataset-v2/save_0",
        help='the path to the decision transformer dataset')

    # Algorithm specific arguments
    parser.add_argument(
        '--games-per-match', 
        type=int,
        default=10,
        help='the number of games to be played in a match'
    )
    parser.add_argument(
        "--agents",
        nargs='+',
        help="the path to the agent models to be ran in the tournament"
    )
    parser.add_argument(
        '--ais', 
        nargs='+',
        help="the ais to be ran in the tournament"
    )
    parser.add_argument(
        '--dts',
        nargs='+',
        help="the path to the decision transformers to be ran in the tournament"
    )
    parser.add_argument(
        '--eval-map',
        type=str,
        default="maps/16x16/basesWorkers16x16A.xml",
        help="the map to be used in the tournament"
    )

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())
    return args


def get_bot_name(bot, bot_type):
    if bot_type == "agent":
         delimiter = "/" if "/" in bot["path"] else "\\"
         return bot["path"].split(delimiter)[-1].split(".")[0]
     elif bot_type == "dt":
         delimiter = "/" if "/" in bot["path"] else "\\"
         split_path = bot["path"].split(delimiter)
         if split_path[-1] == "":
             return split_path[-2]
         else:
             return split_path[-1]
     elif bot_type == "ai":
         return bot["classname"]
     else:
         raise ValueError(f"Unknown bot type: {bot_type}")


def print_final_results(bots_wins, all_bots):
    print("\nFinal Results:")
    bot_names = [get_bot_name(bot, bot["type"]) for bot in all_bots]
    header = " " * 15 + " | " + " | ".join(f"{name:>15}" for name in bot_names) + " | Total Wins"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(bots_wins):
        total_wins = sum(row)
        row_str = f"{bot_names[i]:>15} | " + " | ".join(f"{win:>15}" for win in row) + f" | {total_wins:>10}"
        print(row_str)


if __name__ == "__main__":
    args = parse_args()

    from ppo_gridnet import Agent, MicroRTSStatsRecorder

    from gym_microrts.envs.vec_env import (MicroRTSBotVecEnv,
                                           MicroRTSGridModeVecEnv)

    # TRY NOT TO MODIFY: seeding
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    max_ep_length = 2000
    exp_name = f"tournament-{time.time()}".replace(".", "")
    mapsize = 16 * 16

    # variables for decision transformer
    TARGET_RETURN = 10
    dataset = DatasetDict.load_from_disk(args.dt_dataset)
    collector = DecisionTransformerGymDataCollator(dataset["train"])
    
    all_bots = []
    if args.agents:
        all_bots.extend(
            [{"type": "agent", "path": agent} for agent in args.agents]
        )
    if args.ais:
        all_bots.extend(
            [{"type": "ai", "classname": ai} for ai in args.ais]
        )
    if args.dts:
         all_bots.extend(
             [{"type": "dt", "path": dt} for dt in args.dts]
         )
    assert len(all_bots) > 1, "at least 2 agents/ais are required to play a tournament"

    bots_wins = np.zeros((len(all_bots), len(all_bots)), dtype=np.int32)

    for i in range(len(all_bots)):
        for j in range(i + 1, len(all_bots)):
            player1_type = all_bots[i]["type"]
            player2_type = all_bots[j]["type"]
            ai1s = None
            ai2s = None

            p1_name = get_bot_name(all_bots[i], player1_type)
            p2_name = get_bot_name(all_bots[j], player2_type)

            if player1_type == "agent" and player2_type == "agent":
                bot_envs, selfplay_envs = 0, 2 
                mode = 0

            elif player1_type == "agent" and player2_type == "ai" or \
                player1_type == "ai" and player2_type == "agent":
                bot_envs, selfplay_envs = 1, 0
                mode = 1

                if player1_type == "ai":
                    ai2s = [eval(f"microrts_ai.{all_bots[i]['classname']}")]
                else:
                    ai2s = [eval(f"microrts_ai.{all_bots[j]['classname']}")]
                
            elif player1_type == "ai" and player2_type == "ai":
                bot_envs, selfplay_envs = 1, 0
                mode = 2
                
                ai1s = [eval(f"microrts_ai.{all_bots[i]['classname']}")]
                ai2s = [eval(f"microrts_ai.{all_bots[j]['classname']}")]
                
            elif player1_type == "dt" and player2_type == "agent" or \
                 player1_type == "agent" and player2_type == "dt":
                 bot_envs, selfplay_envs = 0, 2
                 mode = 3
 
            elif player1_type == "dt" and player2_type == "ai" or \
                player1_type == "ai" and player2_type == "dt":
                bot_envs, selfplay_envs = 1, 0
                mode = 4

                if player1_type == "ai":
                    ai2s = [eval(f"microrts_ai.{all_bots[i]['classname']}")]
                else:
                    ai2s = [eval(f"microrts_ai.{all_bots[j]['classname']}")]

            else:
                raise ValueError("DT vs. DT is not currently supported")

            num_envs = bot_envs + selfplay_envs

            if mode == 0:
                envs = MicroRTSGridModeVecEnv(
                    num_bot_envs=bot_envs,
                    num_selfplay_envs=selfplay_envs,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=[args.eval_map],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )

            elif mode == 1:
                envs = MicroRTSGridModeVecEnv(
                    num_bot_envs=bot_envs,
                    num_selfplay_envs=selfplay_envs,
                    ai2s=ai2s,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=[args.eval_map],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )

            elif mode == 2:
                envs = MicroRTSBotVecEnv(
                    ai1s=ai1s,
                    ai2s=ai2s,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=[args.eval_map],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )
            elif mode == 3:
                envs = MicroRTSBotVecEnv(
                    num_bot_envs=bot_envs,
                    num_selfplay_envs=selfplay_envs,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=[args.eval_map],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )
            elif mode == 4:
                envs = MicroRTSBotVecEnv(
                    num_bot_envs=bot_envs,
                    num_selfplay_envs=selfplay_envs,
                    ai2s=ai2s,
                    partial_obs=False,
                    max_steps=max_ep_length,
                    render_theme=2,
                    map_paths=[args.eval_map],
                    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
                    autobuild=False
                )
                
            envs = MicroRTSStatsRecorder(envs)
            envs = VecMonitor(envs)

            if args.capture_video:
                envs = VecVideoRecorder(
                    envs, 
                    f"videos/{exp_name}/{p1_name}-{p2_name}",
                    record_video_trigger=lambda x: x == 0,
                    video_length=max_ep_length,
                )

            if mode == 0:
                agent1 = Agent(envs).to(device)
                agent2 = Agent(envs).to(device)

                agent1.load_state_dict(torch.load(all_bots[i]["path"], map_location=device), strict=False)
                agent1.eval()
                agent2.load_state_dict(torch.load(all_bots[j]["path"], map_location=device), strict=False)
                agent2.eval()

                next_obs = torch.Tensor(envs.reset()).to(device)
                
            elif mode == 1:
                agent1 = Agent(envs).to(device)

                if player1_type == "agent":
                    agent1.load_state_dict(torch.load(all_bots[i]["path"], map_location=device), strict=False)
                else:
                    agent1.load_state_dict(torch.load(all_bots[j]["path"], map_location=device), strict=False)
                agent1.eval()

                next_obs = torch.Tensor(envs.reset()).to(device)

            elif mode == 2:
                next_obs = envs.reset()

            elif mode == 3:
                agent1 = TrainableDT.from_pretrained(all_bots[i]["path"] if player1_type == "dt else all_bots[j]["path"]).to(device)
                agent1.eval()

                agent2 = Agent(envs).to(device)
                agent2.load_state_dict(
                     torch.load(
                         all_bots[i]["path"] if player1_type == "agent" else all_bots[j]["path"],
                         map_location=device
                     ),
                     strict=False
                 )
                 agent2.eval()
 
                 next_obs = torch.Tensor(envs.reset()).to(device)
            elif mode == 4:
                 agent1 = TrainableDT.from_pretrained(all_bots[i]["path"] if player1_type == "dt" else all_bots[j]["path"]).to(device)
                 agent1.eval()
                 
                 next_obs = torch.Tensor(envs.reset()).to(device)
                
                                                     
            print("\n\n====== Next Match ======")
            print(f"{p1_name} vs. {p2_name}")

            for game in range(args.games_per_match):
                # reset the DT trajectory
                 if mode == 3 or mode == 4:
                     states = decode_obs(next_obs[0].view(mapsize, -1)).reshape(
                         1, collector.state_dim).to(device=device, dtype=torch.float32)
                     target_return = torch.tensor(
                         TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
                     actions = torch.zeros(
                         (0, collector.act_dim), device=device, dtype=torch.float32)
                     rewards = torch.zeros(0, device=device, dtype=torch.float32)
                     timesteps = torch.tensor(
                         0, device=device, dtype=torch.long).reshape(1, 1)
                              
                for update in range(max_ep_length):
                    envs.render(mode="rgb_array")
                    if mode == 0:
                        with torch.no_grad():
                             invalid_action_masks = torch.tensor(
                                 np.array(envs.get_action_mask())).to(device)
 
                             p1_obs = next_obs[::2]
                             p2_obs = next_obs[1::2]
                             p1_mask = invalid_action_masks[::2]
                             p2_mask = invalid_action_masks[1::2]
 
                             p1_action, _, _, _, _ = agent1.get_action_and_value(
                                 p1_obs, envs=envs, invalid_action_masks=p1_mask, device=device
                             )
                             p2_action, _, _, _, _ = agent2.get_action_and_value(
                                 p2_obs, envs=envs, invalid_action_masks=p2_mask, device=device
                             )
                             action = torch.zeros(
                                 (bot_envs + selfplay_envs, p2_action.shape[1], p2_action.shape[2]))
                             action[::2] = p1_action
                             action[1::2] = p2_action
 
                             try:
                                 next_obs, _, ds, infos = envs.step(
                                     action.cpu().numpy().reshape(envs.num_envs, -1))
                                 next_obs = torch.Tensor(next_obs).to(device)
                             except Exception as e:
                                 e.printStackTrace()
                                 raise
 
                     elif mode == 1:
                        with torch.no_grad():
                            invalid_action_masks = torch.tensor(
                                np.array(envs.get_action_mask())).to(device)

                                action, _, _, _, _ = agent1.get_action_and_value(
                                 next_obs, envs=envs, invalid_action_masks=invalid_action_masks, device=device
                             )
 
                             try:
                                 next_obs, _, ds, infos = envs.step(
                                     action.cpu().numpy().reshape(envs.num_envs, -1))
                                 next_obs = torch.Tensor(next_obs).to(device)
                             except Exception as e:
                                 e.printStackTrace()
                                 raise
 
                     elif mode == 2:
                        # dummy action
                        next_obs, _, ds, infos = envs.step(
                            [
                                [
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                ]
                            ]
                        )

                     elif mode == 3:
                         with torch.no_grad():
                             invalid_action_masks = torch.tensor(
                                 np.array(envs.get_action_mask())).to(device)
 
                             actions = torch.cat([actions, torch.zeros(
                                 (1, collector.act_dim), device=device)], dim=0)
                             rewards = torch.cat(
                                 [rewards, torch.zeros(1, device=device)])
 
                             p1_action = get_action(
                                 agent1,
                                 (states - collector.state_mean) / collector.state_std,
                                 actions,
                                 target_return,
                                 timesteps,
                                 invalid_action_masks[0]
                             )
                             actions[-1] = p1_action
 
                             action = torch.zeros(
                                 (num_envs, mapsize, 7), device=device)
 
                             action[::2] = decode_action(
                                 p1_action.view(mapsize, -1)).view(1, mapsize, -1)
 
                             p2_obs = next_obs[1::2]
                             p2_mask = invalid_action_masks[1::2]
 
                             p2_action, _, _, _, _ = agent2.get_action_and_value(
                                 p2_obs, envs=envs, invalid_action_masks=p2_mask, device=device
                             )
                             action[1::2] = p2_action
 
                             action = action.detach().cpu().numpy()
 
                             try:
                                 next_obs, rs, ds, infos = envs.step(
                                     action.reshape(envs.num_envs, -1))
                                 next_obs = torch.Tensor(next_obs).to(device)
                             except Exception as e:
                                 e.printStackTrace()
                                 raise
 
                             next_state = torch.Tensor(
                                 decode_obs(next_obs[0].view(mapsize, -1))
                             ).to(device).reshape(1, collector.state_dim)
 
                             states = torch.cat([states, next_state], dim=0)
                             rewards[-1] = rs[0]
 
                             pred_return = target_return[0, -1] - rs[0]
                             target_return = torch.cat(
                                 [target_return, pred_return.reshape(1, 1)], dim=1)
                             timesteps = torch.cat([timesteps, torch.ones(
                                 (1, 1), device=device, dtype=torch.long) * (update + 1)], dim=1)
 
                    elif mode == 4:
                        with torch.no_grad():
                            invalid_action_masks = torch.tensor(
                               np.array(envs.get_action_mask())).to(device)
 
                            actions = torch.cat([actions, torch.zeros(
                                (1, collector.act_dim), device=device)], dim=0)
                            rewards = torch.cat(
                              [rewards, torch.zeros(1, device=device)])
 
                            p1_action = get_action(
                                agent1,
                               (states - collector.state_mean) / collector.state_std,
                                actions,
                                target_return,
                             timesteps,
                             invalid_action_masks[0]
                             )
                            actions[-1] = p1_action
 
                            action = torch.zeros(
                                (num_envs, mapsize, 7), device=device)
 
                            action[::2] = decode_action(
                               p1_action.view(mapsize, -1)).view(1, mapsize, -1)
 
                            action = action.detach().cpu().numpy()
 
                            try:
                                next_obs, rs, ds, infos = envs.step(
                                    action.reshape(envs.num_envs, -1))
                                next_obs = torch.Tensor(next_obs).to(device)
                            except Exception as e:
                                e.printStackTrace()
                                raise

                            next_state = torch.Tensor(
                                decode_obs(next_obs[0].view(mapsize, -1))
                            ).to(device).reshape(1, collector.state_dim)

                         states = torch.cat([states, next_state], dim=0)
                         rewards[-1] = rs[0]

                         pred_return = target_return[0, -1] - rs[0]
                         target_return = torch.cat(
                             [target_return, pred_return.reshape(1, 1)], dim=1)
                         timesteps = torch.cat([timesteps, torch.ones(
                                (1, 1), device=device, dtype=torch.long) * (update + 1)], dim=1)
                                   
                    for idx, info in enumerate(infos):
                        if "episode" in info.keys():
                            score = int(info["microrts_stats"]["WinLossRewardFunction"])

                            if mode == 0:
                                if idx % 2 == 0:
                                    if score > 0:
                                        bots_wins[i][j] += 1
                                        print(f"Game {game + 1}: {p1_name} wins!")
                                    elif score == 0:
                                        print(f"Game {game + 1}: it's a draw!")
                                elif score > 0:
                                    bots_wins[j][i] += 1
                                    print(f"Game {game + 1}: {p2_name} wins!")

                            elif mode in {1, 4}:
                                 if (player1_type == "agent" and mode == 1) or (player1_type == "dt" and mode == 4):
                                    if score > 0:
                                        bots_wins[i][j] += 1
                                        print(f"Game {game + 1}: {p1_name} wins!")
                                    elif score < 0:
                                        bots_wins[j][i] += 1
                                        print(f"Game {game + 1}: {p2_name} wins!")
                                    else:
                                        print(f"Game {game + 1}: it's a draw!")
                                else:
                                    if score > 0:
                                        bots_wins[j][i] += 1
                                        print(f"Game {game + 1}: {p2_name} wins!")
                                    elif score < 0:
                                        bots_wins[i][j] += 1
                                        print(f"Game {game + 1}: {p1_name} wins!")
                                    else:
                                        print(f"Game {game + 1}: it's a draw!")

                            elif mode == 2:
                                if score > 0:
                                    bots_wins[i][j] += 1
                                    print(f"Game {game + 1}: {p1_name} wins!")
                                elif score < 0:
                                    bots_wins[j][i] += 1
                                    print(f"Game {game + 1}: {p2_name} wins!")
                                else:
                                    print(f"Game {game + 1}: it's a draw!")

                            elif mode == 3:
                                 if idx % 2 == 0:
                                     if player1_type == "dt":
                                         if score > 0:
                                             bots_wins[i][j] += 1
                                             print(f"Game {game + 1}: {p1_name} wins!")
                                         elif score == 0:
                                             print(f"Game {game + 1}: it's a draw!")
                                     else:
                                         if score > 0:
                                             bots_wins[j][i] += 1
                                             print(f"Game {game + 1}: {p2_name} wins!")
                                         elif score == 0:
                                             print(f"Game {game + 1}: it's a draw!")
                                 elif score > 0:
                                     if player1_type == "dt":
                                         bots_wins[j][i] += 1
                                         print(f"Game {game + 1}: {p2_name} wins!")
                                     else:
                                         bots_wins[i][j] += 1
                                         print(f"Game {game + 1}: {p1_name} wins!")
                                   
                    # game exit condition
                    if ds[0]:
                        break;

            print("====== Match Done ======\n\n")

    print_final_results(bots_wins, all_bots)
