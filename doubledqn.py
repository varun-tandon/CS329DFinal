import math
import random

import torch
import torch.optim as optim
import torch.nn as nn

from network import DQN
from shared import BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START, LR, device, TAU, Transition, GAMMA
from adversarial import AdversarialAgent

class DoubleDQNAgent():
    def __init__(self, n_observations, n_actions, action_space):
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.action_space = action_space
        self.steps_done = 0

        self.adversary = AdversarialAgent()

    def act(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)
    
    def get_best_action(self, state):
        return self.policy_net(state).max(1)[1].view(1, 1)
    
    def differentiable_reward(state):
        return 

    def optimize_model(self, memory):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        if SHOULD_GENERATE_ADV:
            adversarial_state = torch.clone(state_batch).detach().requires_grad_(True)
            adv_optim = optim.Adam([adversarial_state], lr=1e-1)
            self.policy_net.eval()
            prev_loss = 0
            while True:
                reward = torch.exp(-torch.abs(torch.clamp(adversarial_state[:, 2], -0.418, 0.418)))
                cost = torch.norm(adversarial_state - state_batch, p=2)
                maxQ = torch.clamp(self.policy_net(adversarial_state).max(1)[0], 0, 1)
                loss = reward + ADV_GAMMA * cost + GAMMA * maxQ
                loss = loss.mean()
                if torch.abs(loss - prev_loss) < 1e-2:
                    break
                prev_loss = loss
                adv_optim.zero_grad()
                loss.backward()
                adv_optim.step()

            self.policy_net.train()
            state_batch = adversarial_state

            
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def optimize_robust_model(self, memory):

        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        adversarial_non_final_next_states = self.adversary.example(non_final_next_states, self.policy_net)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(adversarial_non_final_next_states).max(1)[0]

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    # def optimize_model(self, memory):
    #     if len(memory) < BATCH_SIZE:
    #         return
    #     transitions = memory.sample(BATCH_SIZE)
    #     batch = Transition(*zip(*transitions))

    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                         batch.next_state)), device=device, dtype=torch.bool)
    #     non_final_next_states = torch.cat([s for s in batch.next_state
    #                                                 if s is not None])
    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action)
    #     reward_batch = torch.cat(batch.reward)

    #     state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    #     next_state_values = torch.zeros_like(state_action_values, device=device)
    #     with torch.no_grad():
    #         non_final_next_state_actions = self.target_net(non_final_next_states).max(1)[1].unsqueeze(-1)
    #         next_state_values[non_final_mask]  = self.policy_net(non_final_next_states).gather(1, non_final_next_state_actions)

    #     expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #     criterion = nn.SmoothL1Loss()
    #     loss = criterion(state_action_values, expected_state_action_values)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    #     self.optimizer.step()


    def update_target_network_weights(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
