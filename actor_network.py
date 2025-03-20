import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta

"""
Actor Network Class
------------
Creates a neural network with pytorch: observation layer (3x128), hidden layer (128x128), and action layer (128x1).
The network is continuous, whose action space is a single decimal value which in this case controls the vessel's pitch.
"""


class ContinuousActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=128, fc2_dims=128, chkpt_dir='tmp/ppo'):
        super(ContinuousActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_continuous_ppo')
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.alpha = nn.Linear(fc2_dims, n_actions)
        self.beta = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        # alpha = F.softplus(self.alpha(x))  + 1.0
        # beta = F.softplus(self.beta(x))  + 1.0
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        dist = Beta(alpha, beta)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
