import math
import random

import torch
import torch.optim as optim
import torch.nn as nn

from network import DQN
from shared import BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START, LR, device, TAU, Transition, GAMMA, GAMMA_ADV

class AdversarialAgent():
	def __init__(self):
		self.gravity = 9.8
		self.masscart = 1.0
		self.masspole = 0.1
		self.total_mass = self.masspole + self.masscart
		self.length = 0.5  # actually half the pole's length
		self.polemass_length = self.masspole * self.length
		self.force_mag = 10.0
		self.tau = 0.02  # seconds between state updates
		self.kinematics_integrator = "euler"

		# Angle at which to fail the episode
		self.theta_threshold_radians = 12 * 2 * math.pi / 360
		self.x_threshold = 2.4

	def rect(self, value, threshold):
		power = 100
		return torch.max(1 - torch.pow(value/threshold, power),0)[0]

	def reward(self, state, action):
		x, x_dot, theta, theta_dot = state[:,0], state[:,1], state[:,2], state[:,3]
		force = self.force_mag if action == 1 else -self.force_mag
		costheta = torch.cos(theta)
		sintheta = torch.sin(theta)

		# For the interested reader:
		# https://coneural.org/florian/papers/05_cart_pole.pdf
		temp = (
			force + self.polemass_length * theta_dot**2 * sintheta
		) / self.total_mass
		thetaacc = (self.gravity * sintheta - costheta * temp) / (
			self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
		)
		xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

		if self.kinematics_integrator == "euler":
			x = x + self.tau * x_dot
			x_dot = x_dot + self.tau * xacc
			theta = theta + self.tau * theta_dot
			theta_dot = theta_dot + self.tau * thetaacc
		else:  # semi-implicit euler
			x_dot = x_dot + self.tau * xacc
			x = x + self.tau * x_dot
			theta_dot = theta_dot + self.tau * thetaacc
			theta = theta + self.tau * theta_dot

		# computes next_state
		# self.state = (x, x_dot, theta, theta_dot)

		# modified reward
		reward = self.rect(x, self.x_threshold) * self.rect(theta, self.theta_threshold_radians)

		return reward

	def reward_paper(self, state, action):
		x, x_dot, theta, theta_dot = state[:,0], state[:,1], state[:,2], state[:,3]
		return torch.exp(-torch.abs(theta))

	# def adversary(current_state_batch, next_state_batch, Q_net):
	# 	"""
	# 	returns adversarial example
	# 	"""

	# 	batch_size, state_dim = current_state_batch.shape
	# 	adverarial_next_state_list = []

	# 	for i in range(batch_size):
	# 		prior_state = current_state_batch[i]
	# 		next_state    = next_state_batch[i]

	# 		adversarial_state = ...

	# 		ADV_ITER = 1000
	# 		for t in range(ADV_ITER):
	# 			action_values = Q_net(adversarial_state)
	# 			payoff_action_0 = self.reward(prior_state, 0) + GAMMA*action_values[0]
	# 			payoff_action_1 = self.reward(prior_state, 1) + GAMMA*action_values[1]

	# 			loss = torch.max(payoff_action_0, payoff_action_1) \
	# 				   + GAMMA_ADV*torch.linalg.vector_norm(adversarial_state - prior_state)

	# 			# update on adversarial state

	# 		adverarial_next_state_list.append(adversarial_state)


	# 	adversarial_state_batch = torch.cat(adverarial_next_state_list)

	# 	return adversarial_state_batch


	def example(self, next_state_batch, Q_net):
		"""
		returns adversarial example
		"""

		batch_size, state_dim = next_state_batch.shape

		adversarial_state_batch = next_state_batch.detach().clone() + 0.1*torch.randn_like(next_state_batch)
		adversarial_state_batch.requires_grad_()

		optimizer = optim.AdamW([adversarial_state_batch], lr=LR, amsgrad=True)

		ADV_ITER = 5000
		for t in range(ADV_ITER):
			action_values = Q_net(adversarial_state_batch)

			# payoff_action_0 = self.reward(adversarial_state_batch, 0) + GAMMA*action_values[:,0]
			# payoff_action_1 = self.reward(adversarial_state_batch, 1) + GAMMA*action_values[:,1]

			payoff_action_0 = self.reward_paper(adversarial_state_batch, 0) + GAMMA*action_values[:,0]
			payoff_action_1 = self.reward_paper(adversarial_state_batch, 1) + GAMMA*action_values[:,1]

			loss = torch.mean(torch.maximum(payoff_action_0, payoff_action_1)) \
				   + GAMMA_ADV*torch.linalg.vector_norm(adversarial_state_batch - next_state_batch)			
			
			if t % 100 == 0:
				print(loss)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

		print(next_state_batch[0])
		print(adversarial_state_batch[0])
		print(torch.linalg.vector_norm(adversarial_state_batch - next_state_batch))
		exit()

		return adversarial_state_batch.detach()


















