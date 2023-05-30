from collections import namedtuple
import torch

SHOULD_PLOT = True
SHOULD_GENERATE_ADV = False 
ADV_GAMMA = 1
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
GAMMA_ADV= 0.5
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 600
