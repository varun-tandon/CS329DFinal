from collections import namedtuple
import torch

SHOULD_PLOT = True 
SHOULD_GENERATE_ADV = True 
GAMMA_ADV = 1 
BATCH_SIZE = 64 
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TAU = 0.005
LR = 1e-4
SELECTED_AGENT = 'tabular'
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 10000

# Test Harness Parameters
TEST_HARNESS_NUM_EPISODES = 30
