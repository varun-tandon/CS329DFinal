import numpy as np
import torch
from shared import SHOULD_GENERATE_ADV, GAMMA_ADV

class QLearningAgent:
    def __init__(self, n_observations, n_actions, action_space):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.action_space = action_space
        self.N_BUCKETS_ANGLE = 30
        self.N_BUCKETS_ANGLE_VEL = 15 
        self.is_exploration_disabled = False

        self.Q = np.zeros((self.N_BUCKETS_ANGLE, self.N_BUCKETS_ANGLE_VEL, self.n_actions))
        self.alpha = 0.5
        self.alpha_decay = 0.9995
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.n_iter = 0
    
    def discretize(self, obs):
        # our observation is a 4-tuple of floats: (x, x_dot, theta, theta_dot)
        # we only care about the angle (theta) and angular velocity (theta_dot)
        # so we will use a simple binning method to discretize these values
        # into buckets
        # the angle is bounded by +/- 0.2095 radians (12 degrees)
        # we can bin the velocity between +/- 1 radians per second, with the first and last
        # buckets representing any values outside of this range
        # we will use 30 bins for the angle and 15 for the angular velocity

        # angle
        angle = obs[2]
        angle_bounds = [-0.2095, 0.2095]
        angle_idx = np.digitize(angle, np.linspace(angle_bounds[0], angle_bounds[1], self.N_BUCKETS_ANGLE))
        if angle_idx == self.N_BUCKETS_ANGLE:
            angle_idx -= 1

        # angular velocity
        angle_vel = obs[3]
        angle_vel_bounds = [-1, 1]
        angle_vel_idx = np.digitize(angle_vel, np.linspace(angle_vel_bounds[0], angle_vel_bounds[1], self.N_BUCKETS_ANGLE_VEL))
        if angle_vel_idx == self.N_BUCKETS_ANGLE_VEL:
            angle_vel_idx -= 1

        return angle_idx, angle_vel_idx

    def act(self, state):
        angle_idx, angle_vel_idx = self.discretize(state[0])
        if np.random.random() < self.epsilon and not self.is_exploration_disabled:
            return self.action_space.sample()
        else:
            return np.argmax(self.Q[angle_idx][angle_vel_idx])

    def disable_exploration(self):
        self.is_exploration_disabled = True

    def optimize_model(self, memory):
        state, action, next_state, reward = memory.get_last()
        if next_state is None:
            self.epsilon *= self.epsilon_decay
            self.alpha *= self.alpha_decay
            print("Epsilon: {}, Alpha: {}".format(self.epsilon, self.alpha))
            self.n_iter += 1
            return 
        angle_idx, angle_vel_idx = self.discretize(state[0])
        original_next_angle_idx, original_next_angle_vel_idx = self.discretize(next_state[0])
        if SHOULD_GENERATE_ADV and self.n_iter % 1 == 0:
            adversarial_state = torch.clone(state).detach().requires_grad_(True)
            adv_optim = torch.optim.Adam([adversarial_state], lr=1e-2)
            prev_loss = 0
            n_iter = 0
            while True:
                reward_adv = torch.exp(-torch.abs(adversarial_state[0][2]))
                reward_next_state = torch.exp(-torch.abs(next_state[0][2]))
                cost = torch.norm(adversarial_state - next_state, p=2)
                loss = reward_adv + GAMMA_ADV * cost
                if torch.abs(loss - prev_loss) < 1e-3:
                    # print("Converged in {} iterations".format(n_iter))
                    break
                prev_loss = loss
                adv_optim.zero_grad()
                loss.backward()
                adv_optim.step()
                n_iter += 1
            next_state = adversarial_state.detach()
        next_angle_idx, next_angle_vel_idx = self.discretize(next_state[0])
        # print(original_next_angle_idx, original_next_angle_vel_idx, next_angle_idx, next_angle_vel_idx)
        max_Q = np.max(self.Q[next_angle_idx][next_angle_vel_idx])
        prev_Q = self.Q[angle_idx][angle_vel_idx][action]
        self.Q[angle_idx][angle_vel_idx][action] += self.alpha * (reward + self.gamma * max_Q - prev_Q) 

    def save(self, path):
        np.save(path + ".npy", self.Q)

    def load(self, path):
        self.Q = np.load(path + ".npy")
