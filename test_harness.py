import gymnasium as gym
import numpy as np
import custom_envs
import json
from tqdm import tqdm
from doubledqn import DoubleDQNAgent
from replaymemory import ReplayMemory
from itertools import count
import torch
from shared import device, TEST_HARNESS_NUM_EPISODES

env_name = 'custom_envs/CartPole-v1'

# gravity_sweep = np.linspace(0.5, 20.0, 5)
# gravity_sweep[2] = 9.8
# gravity_sweep = list(gravity_sweep)
gravity_sweep = [9.8 / 5, 9.8, 9.8 * 5]

# try masses from 0.1 to 2.0 in increments of 0.1
# masscart_sweep = np.linspace(0.1, 2.0, 5)
# masscart_sweep[2] = 1.0
# masscart_sweep = list(masscart_sweep)
masscart_sweep = [1.0 / 2, 1.0, 1.0 * 2]

# try masses from 0.01 to 0.2 in increments of 0.01
# masspole_sweep = np.linspace(0.01, 0.2, 5)
# masspole_sweep[2] = 0.1
# masspole_sweep = list(masspole_sweep)
masspole_sweep = [0.1 / 2, 0.1, 0.1 * 2]

# try lengths from 0.1 to 2.0 in increments of 0.1
# length_sweep = np.linspace(0.1, 1.0, 5)
# length_sweep[2] = 0.5
# length_sweep = list(length_sweep)
length_sweep = [0.5 / 2, 0.5, 0.5 * 2]

results = dict()

all_combinations = []
for gravity in gravity_sweep:
    for masscart in masscart_sweep:
        for masspole in masspole_sweep:
            for length in length_sweep:
                all_combinations.append((gravity, masscart, masspole, length))


for gravity, masscart, masspole, length in tqdm(all_combinations):
    print(f'Running test harness for gravity={gravity}, masscart={masscart}, masspole={masspole}, length={length}')
    episode_durations = []
    options = {
        'gravity': gravity,
        'masscart': masscart,
        'masspole': masspole,
        'length': length
    }
    env = gym.make(env_name, options=options)
    n_actions = env.action_space.n
    for _ in range(TEST_HARNESS_NUM_EPISODES):
        state, info = env.reset()
        n_observations = len(state)
    
        agent = DoubleDQNAgent(n_observations, n_actions, env.action_space)
        agent.load_model('models\\CartPole-v1-232-episodes-1685491412.3060522.pth')
        agent.disable_exploration()
        
        for t in count():
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if done:
                break
        episode_durations.append(t + 1)

    mean = np.mean(episode_durations)
    std = np.std(episode_durations)
    gravity = float(gravity)
    masscart = float(masscart)
    masspole = float(masspole)
    length = float(length)
    results[f'{gravity},{masscart},{masspole},{length}'] = {'mean': mean, 'std': std}
    print(f'gravity={gravity}, masscart={masscart}, masspole={masspole}, length={length}, mean={mean}, std={std}')
    env.close()

json.dump(results, open('results_paper.json', 'w'))

