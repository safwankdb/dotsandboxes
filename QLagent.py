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
import numpy as np
from models import DQN

logger = logging.getLogger(__name__)
games = {}

EPS_START = 1
DECAY_LEN = 50_000
EPS_END = 0.1
init = True
agent = None

class QLEnv:

    def __init__(self, player, nb_rows, nb_cols, timelimit, episode):

        self.EPSILON = EPS_END + (EPS_START - EPS_END)*(1-episode/DECAY_LEN)
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
        self.dqn = DQN(self.len_states, self.len_states)

    def reset(self, episode):
        self.episode = episode
        self.EPSILON = EPS_END + (EPS_START - EPS_END)*(1-episode/DECAY_LEN)
        self.EPSILON = max(self.EPSILON, EPS_END)
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
        if self.episode + 1 % 100_000 == 0:
            torch.save(self.dqn.model.state_dict(), f"model_{self.nb_rows}_rows_{self.nb_cols}_cols_{episode+1}.pth")


    def process_next_state(self, score):
        if self.player == 2:
            score = score[::-1]
        self.reward = score[0] - self.score[0] - score[1] + self.score[1]
        # self.reward /= 100
        self.score = score
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
            self.reward += 100
        elif self.score[0] < self.score[1]:
            self.reward += -100
        self.ended = True
        self.dqn.memorize(self.prev_state, self.action,
                          self.reward, self.state, done=True)
        self.dqn.train(terminal=True)


# MAIN EVENT LOOP

async def handler(websocket, path):
    global init, agent
    logger.info("Start listening")
    # msg = await websocket.recv()
    async for msg in websocket:
        logger.info("< {}".format(msg))
        msg = json.loads(msg)
        answer = None
        if msg["type"] == "start":
            # Initialize game
            nb_rows, nb_cols = msg["grid"]
            if init:
                agent = QLEnv(msg["player"], nb_rows, nb_cols,
                            msg["timelimit"], msg["episode"])
                init = False
            else:
                agent.reset(msg["episode"])
            if msg["player"] == 1:
                # Start the game
                nm = agent.next_action()
                if nm is None:
                    # Game over
                    logger.info("Game over")
                    continue
                r, c, o = nm
                answer = {
                    'type': 'action',
                    'location': [r, c],
                    'orientation': o
                }
            else:
                # Wait for the opponent
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


def start_server(port):
    server = websockets.serve(handler, 'localhost', port)
    print("Running on ws://127.0.0.1:{}".format(port))
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
    args = parser.parse_args(argv)
    logger.setLevel(
        max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    start_server(args.port)


if __name__ == "__main__":
    sys.exit(main())
