import os
import torch as T
import torch.nn as nn
import torch.optim as optim

"""
Critic Network Class
------------
Creates a neural network with pytorch: observation layer (3x128), hidden layer (128x128), and critic layer (128x1).
The network is continuous, and it's critic value discerns quality of the actor network's previous action or policy.
The advantage of an Actor-Critic method, where actor and critic networks work in tandem, is that the critic network's 
discernments dampen the potentially severe updates that could be made to the network which would cause perturbation. 
"""


class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=128, fc2_dims=128, chkpt_dir='tmp/ppo'):
        super(ContinuousCriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_continuous_ppo')
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
