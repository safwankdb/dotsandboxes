#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxescompete.py

Template for the Machine Learning Project course at KU Leuven (2017-2018)
of Hendrik Blockeel and Wannes Meert.

Copyright (c) 2018 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
import asyncio
import websockets
import json
from collections import defaultdict
import random
import uuid
import time
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


def start_competition(address1, address2, nb_rows, nb_cols, timelimit, episodes):
   episode = 0
   winners = []
   for episode in range(episodes):
        swap = random.random() > 0.5
        if swap:
            a1 = address2
            a2 = address1
        else:
            a1 = address1
            a2 = address2
        asyncio.get_event_loop().run_until_complete(connect_agent(a1, a2, nb_rows, nb_cols, timelimit, winners, episode, swap))
        last_n = winners[-min(500, len(winners)):]
        print("Epsiode {} Cumulative Score: {} - {} - {}".format(episode, last_n.count(1), last_n.count(2), last_n.count(0)))
   print("Player 1 won {} times, Player 2 won {} times and Draw occured {} times".format(winners.count(1), winners.count(2), winners.count(0)))


async def connect_agent(uri1, uri2, nb_rows, nb_cols, timelimit, winners, episode, sort_key):
    cur_game = str(uuid.uuid4())
    winner = None
    cells = []
    cur_player = 1
    points = [0, 0, 0]
    timings = [None, [], []]

    for ri in range(nb_rows + 1):
        columns = []
        for ci in range(nb_cols + 1):
            columns.append({"v":0, "h":0, "p":0})
        cells.append(columns)

    async with websockets.connect(uri1) as websocket1:
        async with websockets.connect(uri2) as websocket2:

            msg = {
              "type": "start",
              "episode": episode,
              "player": 1,
              "timelimit": timelimit,
              "game": cur_game,
              "grid": [nb_rows, nb_cols]
            }
            await websocket1.send(json.dumps(msg))
            msg["player"] = 2
            await websocket2.send(json.dumps(msg))

            while winner is None:
                ask_time = time.time()
                if cur_player == 1:
                    msg = await websocket1.recv()
                else:
                    msg = await websocket2.recv()
                recv_time = time.time()
                diff_time = recv_time - ask_time
                timings[cur_player].append(diff_time)
                try:
                    msg = json.loads(msg)
                except json.decoder.JSONDecodeError as err:
                    logger.debug(err)
                    continue
                if msg["type"] != "action":
                    logger.error("Unknown message: {}".format(msg))
                    continue
                r, c = msg["location"]
                o = msg["orientation"]
                next_player = user_action(r, c, o, cur_player,
                                          cells, points,
                                          nb_rows, nb_cols)
                if points[1] + points[2] == nb_cols * nb_rows:
                    winner = 1
                    if points[2] == points[1]:
                        winner = 0
                    if points[2] > points[1]:
                        winner = 2
                else:
                    msg = {
                        "type": "action",
                        "game": cur_game,
                        "player": cur_player,
                        "nextplayer": next_player,
                        "score": [points[1], points[2]],
                        "location": [r, c],
                        "orientation": o
                    }
                    await websocket1.send(json.dumps(msg))
                    await websocket2.send(json.dumps(msg))

                cur_player = next_player

            msg = {
                "type": "end",
                "game": cur_game,
                "player": cur_player,
                "nextplayer": 0,
                "score": [points[1], points[2]],
                "location": [r, c],
                "orientation": o,
                "winner": winner
            }
            await websocket1.send(json.dumps(msg))
            await websocket2.send(json.dumps(msg))

    if sort_key == 0:
        winners.append(winner)
    else:
        winners.append({0:0,1:2,2:1}[winner])
    return winners


def user_action(r, c, o, cur_player, cells, points, nb_rows, nb_cols):
    next_player = cur_player
    won_cell = False
    cell = cells[r][c]
    if o == "h":
        if cell["h"] != 0:
            return cur_player
        cell["h"] = cur_player
        # Above
        if r > 0:
            if cells[r - 1][c]["v"] != 0 \
                and cells[r - 1][c + 1]["v"] != 0 \
                and cells[r - 1][c]["h"] != 0 \
                and cells[r][c]["h"] != 0:
                won_cell = True
                points[cur_player] += 1
                cells[r - 1][c]["p"] = cur_player
        # Below
        if r < nb_rows:
            if cells[r][c]["v"] != 0 \
                and cells[r][c + 1]["v"] != 0 \
                and cells[r][c]["h"] != 0 \
                and cells[r + 1][c]["h"] != 0:
                won_cell = True
                points[cur_player] += 1
                cells[r][c]["p"] = cur_player

    if o == "v":
        if cell["v"] != 0:
            return cur_player
        cell["v"] = cur_player
        # Left
        if c > 0:
            if cells[r][c - 1]["v"] != 0 \
                and cells[r][c]["v"] != 0 \
                and cells[r][c - 1]["h"] != 0 \
                and cells[r + 1][c - 1]["h"] != 0:
                won_cell = True
                points[cur_player] += 1
                cells[r][c - 1]["p"] = cur_player
        # Right
        if c < nb_cols:
            if cells[r][c]["v"] != 0 \
                and cells[r][c + 1]["v"] != 0 \
                and cells[r][c]["h"] != 0 \
                and cells[r + 1][c]["h"] != 0:
                won_cell = True
                points[cur_player] += 1
                cells[r][c]["p"] = cur_player

    if not won_cell:
        next_player = 3 - cur_player
    else:
        next_player = cur_player
        # print("Update points: player1={} - player2={}".format(points[1], points[2]))
    return next_player


def main(argv=None):
    parser = argparse.ArgumentParser(description='Start agent to play Dots and Boxes')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('--cols', '-c', type=int, default=3, help='Number of columns')
    parser.add_argument('--rows', '-r', type=int, default=3, help='Number of rows')
    parser.add_argument('--timelimit', '-t', type=float, default=50, help='Time limit per request in seconds')
    parser.add_argument('--episodes', '-e', type=int, default=200, help='Number of episodes')
    parser.add_argument('agents', nargs=2, metavar='AGENT', help='Websockets addresses for agents')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    start_competition(args.agents[0], args.agents[1], args.rows, args.cols, args.timelimit, args.episodes)


if __name__ == "__main__":
    sys.exit(main())

