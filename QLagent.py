"""
EE782 @ IITB
Course Project

kdbeatbox@gmail.com
"""

import sys
import argparse
import logging
import asyncio
import websockets
import json
from collections import defaultdict
import random

import torch
from torch import nn
import numpy as np
from models import DQN

logger = logging.getLogger(__name__)
games = {}

EPS_START = 1
DECAY_LEN = 5_000
EPS_END = 0.05
SAVE_EVERY = 5_000
init = True
agent = None
test = False


class QLEnv:

    def __init__(self, player, nb_rows, nb_cols, timelimit, episode):

        self.EPSILON = EPS_END + (EPS_START - EPS_END)*(1-(episode/DECAY_LEN))
        self.EPSILON = max(self.EPSILON, EPS_END)
        self.timelimit = timelimit
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        rows = []
        for _ in range(nb_rows + 1):
            columns = []
            for _ in range(nb_cols + 1):
                columns.append({"v": 0, "h": 0})
            rows.append(columns)
        self.cells = rows
        self.len_states = nb_rows*(nb_cols+1)+nb_cols*(nb_rows+1)
        self.state = np.zeros(self.len_states)
        self.player = player
        self.score = [0, 0]
        self.reward = 0
        self.prev_state = None
        self.dqn = DQN(self.len_states, self.len_states)

    def reset(self, episode):
        self.episode = episode
        self.EPSILON = EPS_END + (EPS_START - EPS_END)*(1-(episode/DECAY_LEN))
        self.EPSILON = max(self.EPSILON, EPS_END)
        self.reward = 0
        self.state = np.zeros(self.len_states)
        self.prev_state = None
        self.score = [0, 0]
        rows = []
        for _ in range(self.nb_rows + 1):
            columns = []
            for _ in range(self.nb_cols + 1):
                columns.append({"v": 0, "h": 0})
            rows.append(columns)
        self.cells = rows
        if (self.episode + 1) % SAVE_EVERY == 0:
            torch.save(self.dqn.model.state_dict(),
                       f"model_self_play_{self.nb_rows}_rows_{self.nb_cols}_cols_{episode+1}.pth")

    def process_next_state(self, score):
        if self.player == 2:
            score = score[::-1]
        self.reward = score[0] - self.score[0] - score[1] + self.score[1]
        self.reward *= 100
        self.score = score
        if self.prev_state is None:
            return
        self.dqn.memorize(self.prev_state, self.action,
                          self.reward, self.state)
        self.dqn.train()

    def update_state(self, update_prev=False):
        if update_prev:
            self.prev_state = self.state.copy()
        i = 0
        for ri in range(self.nb_rows):
            for ci in range(self.nb_cols+1):
                value = self.cells[ri][ci]['v']
                if value == self.player:
                    value = 1
                elif value != 0:
                    value = -1
                self.state[i] = value
                i += 1
        for ri in range(self.nb_rows+1):
            for ci in range(self.nb_cols):
                value = self.cells[ri][ci]['h']
                if value == self.player:
                    value = 1
                elif value != 0:
                    value = -1
                self.state[i] = value
                i += 1

    def register_action(self, row, column, orientation, player):
        self.cells[row][column][orientation] = player
        self.update_state(player == self.player)

    def next_action(self):
        free_lines = [i for i in range(len(self.state)) if self.state[i] == 0]
        if len(free_lines) == 0:
            print('end')
            return None
        if np.random.random() > self.EPSILON:
            moves = np.argsort(self.dqn.predict(self.state))
            idx = len(moves) - 1
            while moves[idx] not in free_lines:
                idx -= 1
            movei = moves[idx]
            movei = int(movei)
        else:
            movei = np.random.choice(free_lines)
            movei = int(movei)
        self.action = movei
        if movei < (self.nb_cols+1)*self.nb_rows:
            o = 'v'
            r = movei // (self.nb_cols+1)
            c = movei % (self.nb_cols+1)
        else:
            movei -= (self.nb_cols+1)*self.nb_rows
            o = 'h'
            r = movei // (self.nb_cols)
            c = movei % (self.nb_cols)
        return r, c, o

    def end_game(self):
        if self.score[0] > self.score[1]:
            self.reward += 1000
        elif self.score[0] < self.score[1]:
            self.reward -= 1000
        self.ended = True
        self.dqn.memorize(self.prev_state, self.action,
                          self.reward, self.state, done=True)
        self.dqn.train(terminal=True)


device = 'cpu'


class QLPlayer:
    def create_model(self):
        n = 24
        model = nn.Sequential(
            nn.Linear(self.len_states, n),
            # nn.BatchNorm1d(n),
            nn.ReLU(),
            nn.Linear(n, 2*n),
            # nn.BatchNorm1d(2*n),
            nn.ReLU(),
            nn.Linear(2*n, n),
            # nn.BatchNorm1d(n),
            nn.ReLU(),
            nn.Linear(n, self.len_states),
        )
        return model

    def __init__(self, player, nb_rows, nb_cols, timelimit):
        print(f"Running on device: {device.upper()}")
        self.ended = False
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        rows = []
        for _ in range(nb_rows + 1):
            columns = []
            for _ in range(nb_cols + 1):
                columns.append({"v": 0, "h": 0})
            rows.append(columns)
        self.cells = rows
        self.len_states = nb_rows*(nb_cols+1)+nb_cols*(nb_rows+1)
        self.state = np.zeros(self.len_states)
        self.player = player
        self.model = self.create_model().to(device)
        self.model.load_state_dict(torch.load('model_2_rows_2_cols_20000.pth'))
        self.model.eval()

    def reset(self):
        self.reward = 0
        self.state = np.zeros(self.len_states)
        self.score = [0, 0]
        rows = []
        for _ in range(self.nb_rows + 1):
            columns = []
            for _ in range(self.nb_cols + 1):
                columns.append({"v": 0, "h": 0})
            rows.append(columns)
        self.cells = rows

    def process_next_state(self, score):
        pass

    def update_state(self):
        i = 0
        for ri in range(self.nb_rows):
            for ci in range(self.nb_cols+1):
                value = self.cells[ri][ci]['v']
                if value == self.player:
                    value = 1
                elif value != 0:
                    value = -1
                self.state[i] = value
                i += 1
        for ri in range(self.nb_rows+1):
            for ci in range(self.nb_cols):
                value = self.cells[ri][ci]['h']
                if value == self.player:
                    value = 1
                elif value != 0:
                    value = -1
                self.state[i] = value
                i += 1

    def register_action(self, row, column, orientation, player):
        self.cells[row][column][orientation] = player
        self.update_state()

    def next_action(self):
        free_lines = [i for i in range(len(self.state)) if self.state[i] == 0]
        if len(free_lines) == 0:
            return None
        with torch.no_grad():
            x = torch.Tensor(self.state).unsqueeze(0).to(device)
            out = self.model(x)[0].cpu()
        moves = np.argsort(out)
        idx = len(moves) - 1
        while moves[idx] not in free_lines:
            idx -= 1
        movei = moves[idx]
        movei = int(movei)
        if movei < (self.nb_cols+1)*self.nb_rows:
            o = 'v'
            r = movei // (self.nb_cols+1)
            c = movei % (self.nb_cols+1)
        else:
            movei -= (self.nb_cols+1)*self.nb_rows
            o = 'h'
            r = movei // (self.nb_cols)
            c = movei % (self.nb_cols)
        return r, c, o

    def end_game(self):
        self.ended = True


# MAIN EVENT LOOP

async def handler(websocket, path):
    global init, agent, test
    logger.info("Start listening")
    # msg = await websocket.recv()
    async for msg in websocket:
        logger.info("< {}".format(msg))
        msg = json.loads(msg)
        answer = None
        if msg["type"] == "start":
            nb_rows, nb_cols = msg["grid"]
            if init:
                if not test:
                    agent = QLEnv(msg["player"], nb_rows, nb_cols,
                                  msg["timelimit"], msg["episode"])
                else:
                    agent = QLPlayer(msg["player"], nb_rows,
                                     nb_cols, msg["timelimit"])
                init = False
            else:
                if not test:
                    agent.reset(msg["episode"])
                else:
                    agent.reset()
            if msg["player"] == 1:
                nm = agent.next_action()
                if nm is None:
                    logger.info("Game over")
                    continue
                r, c, o = nm
                answer = {
                    'type': 'action',
                    'location': [r, c],
                    'orientation': o
                }
            else:
                answer = None

        elif msg["type"] == "action":
            r, c = msg["location"]
            o = msg["orientation"]
            agent.register_action(r, c, o, msg["player"])
            if msg["nextplayer"] == agent.player:
                agent.process_next_state(msg['score'])
                nm = agent.next_action()
                if nm is None:
                    logger.info("Game over")
                    continue
                nr, nc, no = nm
                answer = {
                    'type': 'action',
                    'location': [nr, nc],
                    'orientation': no
                }
            else:
                answer = None

        elif msg["type"] == "end":
            # End the game
            agent.end_game()
            answer = None
        else:
            logger.error("Unknown message type:\n{}".format(msg))

        if answer is not None:
            await websocket.send(json.dumps(answer))
            logger.info("> {}".format(answer))
    logger.info("Exit handler")


def start_server(port, test_bool):
    global test
    test = test_bool
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
    print(f"Testing: {test}")
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()


# COMMAND LINE INTERFACE

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Start agent to play Dots and Boxes')
    parser.add_argument('--verbose', '-v', action='count',
                        default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count',
                        default=1, help='Quiet output')
    parser.add_argument('port', metavar='PORT', type=int,
                        help='Port to use for server')
    parser.add_argument('--test', action='store_true', help='Test mode')
    args = parser.parse_args(argv)
    logger.setLevel(
        max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    start_server(args.port, args.test)


if __name__ == "__main__":
    sys.exit(main())
