import math
import random
import time
from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
import custom_envs
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from doubledqn import DoubleDQNAgent
from plots import plot_durations
from replaymemory import ReplayMemory
from shared import device, NUM_EPISODES, SHOULD_PLOT, SHOULD_GENERATE_ADV
from tqdm import tqdm

convergence_times = []

for _ in tqdm(range(20)):
    env_name = "CartPole-v1"
    env = gym.make("custom_envs/CartPole-v1")
    
    if SHOULD_PLOT:
        plt.ion()
    
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)
    
    memory = ReplayMemory(10000)
    agent = DoubleDQNAgent(n_observations, n_actions, env.action_space)
    
    episode_durations = []
    
    has_converged = False
    
    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.act(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
    
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
            memory.push(state, action, next_state, reward)
    
            state = next_state
    
            agent.optimize_model(memory)
            agent.update_target_network_weights()
    
            if done:
                episode_durations.append(t + 1)
                print(f"Episode {i_episode} finished after {t+1} timesteps")
                # if the last 3 episodes have been at 500 timesteps, we have converged
                if len(episode_durations) >= 5 and all([x == 500 for x in episode_durations[-5:]]):
                    print("Converged!")
                    has_converged = True
                if SHOULD_PLOT:
                    plot_durations(episode_durations)
                break
        if has_converged:
            convergence_times.append(i_episode)
            break
    if not has_converged:
        convergence_times.append(-1)
    if SHOULD_PLOT:
        plot_durations(episode_durations, show_result=True)
        plt.ioff()
    print(convergence_times)
    # save the model
    torch.save(agent.policy_net.state_dict(), f"models/{env_name}-{'ADV' if SHOULD_GENERATE_ADV else 'NOADV'}-{i_episode}-episodes-{time.time()}.pth")
print(convergence_times)
