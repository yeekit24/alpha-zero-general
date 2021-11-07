import Arena
from MCTS import MCTS
# from othello.OthelloGame import OthelloGame as Game
# from othello.OthelloPlayers import *
# from othello.pytorch.NNet import NNetWrapper as NNet
from _chess.ChessGame import ChessGame as Game
from _chess.ChessPlayers import *
from _chess.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = Game()
# all players
rp = RandomPlayer(g).play
rp2 = RandomPlayer(g).play

hp = HumanChessPlayer(g).play
sp = StockFishPlayer(g, 2500, 10, 10).play
sp2 = StockFishPlayer(g, 100, 1, 1).play

# nnet players
n1 = NNet(g)

n1.load_checkpoint('./pretrained_models/_chess/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 100, 'cpuct':1.0, 'dir_noise': False})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=1))

n2 = NNet(g)
n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
args2 = dotdict({'numMCTSSims': 100, 'cpuct': 1.0, 'dir_noise': False})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=1))

arena = Arena.Arena(n1p, n2p, g, display=Game.display)
print(arena.playGames(n1p, n2p, 20, verbose=False))
