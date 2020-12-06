import torch
import torch.nn as nn
import numpy as np

import random
from tqdm import tqdm

REPLAY_MEMORY_SIZE = 200
WARMUP_SIZE = 50
GAMMA = 0.9
TARGET_UPDATE = 5
BATCH_SIZE = 32
EPISODES = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nRunning on {device.upper()}\n")

class ExperienceReplay:

    def __init__(self, size=REPLAY_MEMORY_SIZE):
        self.size = size
        self.buffer = []
        self.pointer = 0

    def push(self, s, a, r, s_, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pointer] = (s, a, r, s_, done)
        self.pointer = (self.pointer + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):

    def create_model(self):
        if self.n_states < 16:
            n = 18
        else:
            n = 32
        model = nn.Sequential(
            nn.Linear(self.n_states, n),
            nn.BatchNorm1d(n),
            nn.LeakyReLU(),
            nn.Linear(n, self.n_actions),
            # nn.Tanh(),
        )
        return model

    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_memory = ExperienceReplay()
        self.model = self.create_model().to(device)
        self.target_model = self.create_model().to(device)
        self.target_model.eval()
        self.opt1 = torch.optim.Adam(self.model.parameters())
        self.opt2 = torch.optim.Adam(self.target_model.parameters())
        self.loss = nn.SmoothL1Loss()
        self.target_counter = 0


    def memorize(self, s, a, r, s_, done=False):
        self.replay_memory.push(s, a, r, s_, done)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = torch.Tensor(x).unsqueeze(0).to(device)
            out = self.model(x)[0].cpu()
        return out

    def forward(self, x):
        out = self.model(x)
        return out

    def train(self, terminal=False):
        if len(self.replay_memory.buffer) < WARMUP_SIZE:
            return
        minibatch = self.replay_memory.sample(BATCH_SIZE)
        S = torch.Tensor([i[0] for i in minibatch]).to(device)
        S_ = torch.Tensor([i[3] for i in minibatch]).to(device)

        self.model.eval()
        with torch.no_grad():
            Q = self.model(S)
            Q_ = self.target_model(S_)

        X = []
        y = []
        for i, (s, a, r, _, done) in enumerate(minibatch):
            if done:
                q_ = r
            else:
                q_ = r + GAMMA * Q_[i].max()
            q = Q[i]
            q[a] = q_
            X.append(s)
            y.append(q)
        
        X = torch.Tensor(X).to(device)
        y = torch.stack(y).to(device)

        self.model.train()
        out = self.model(X)
        loss = self.loss(out, y)
        self.opt1.zero_grad()
        loss.backward()
        self.opt1.step()

        if terminal:
            self.target_counter += 1
            if self.target_counter >= TARGET_UPDATE:
                self.target_model.load_state_dict(self.model.state_dict())
            self.target_counter = 0


