import torch
import torch.nn as nn
import numpy as np

import random
from tqdm import tqdm

REPLAY_MEMORY_SIZE = 20_000
WARMUP_SIZE = 5000
GAMMA = 0.9
TARGET_UPDATE = 50
BATCH_SIZE = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
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


class ScaleLayer(nn.Module):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        self.alpha = 1/(1-GAMMA)

    def forward(self, x):
        return self.alpha * x

def create_model(ins, outs):
    if ins <= 12:
        n = 32
    else:
        n = 64
    model = nn.Sequential(
        nn.Linear(ins, n),
        nn.ReLU(),
        nn.Linear(n, 4*n),
        nn.ReLU(),
        nn.Linear(4*n, n),
        nn.ReLU(),
        nn.Linear(n, outs),
        nn.Tanh(),
        ScaleLayer(),
    )
    return model

class DQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_memory = ExperienceReplay()
        self.model = create_model(self.n_states, n_actions).to(device)
        self.target_model = create_model(self.n_states, n_actions).to(device)
        self.target_model.eval()
        self.opt = torch.optim.RMSprop(self.model.parameters(), 1e-3)
        # self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
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
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if terminal:
            self.target_counter += 1
            if self.target_counter >= TARGET_UPDATE:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_counter = 0
