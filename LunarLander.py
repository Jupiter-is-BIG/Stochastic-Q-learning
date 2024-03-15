import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, lr, state_actions, layer_1_dim, layer_2_dim, num_actions):
        super(DQN, self).__init__()
        self.lr = lr
        self.state_actions = state_actions
        self.layer_1_dim = layer_1_dim
        self.layer_2_dim = layer_2_dim
        self.num_actions = num_actions

        self.layer_1 = nn.Linear(*self.state_actions, self.layer_1_dim)
        self.layer_2 = nn.Linear(self.layer_1_dim, self.layer_2_dim)
        self.layer_3 = nn.Linear(self.layer_2_dim, num_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = self.layer_1(state)
        x = nn.functional.relu(x)
        x = self.layer_2(x)
        x = nn.functional.relu(x)
        return self.layer_3(x)
