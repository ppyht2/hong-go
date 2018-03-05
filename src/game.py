import os
import six
import time
import joblib
import pachi_py
import numpy as np


class Player():
    def __init__(self, colour):
        self.colour = colour

    def act(self, state):
        assert state.color == self.colour
        la = self.get_legal_actions(state.board, self.colour)
        a = np.random.choice(la)
        return a

    def get_legal_actions(self, board, color):
        cs = board.get_legal_coords(color)
        a = [_coord_to_action(board, c) for c in cs]
        return a


class PachiPlayer():
    def __init__(self, colour, board):
        self.colour = colour
        self.policy = make_pachi_policy(
            board.clone(), engine_type=six.b('uct'), pachi_timestr=six.b('_2400'))

    def act(self, curr_state, prev_stae, prev_action):
        assert curr_state.color == self.colour
        a = self.policy(curr_state, prev_stae, prev_action)
        return a


class TrainingEnvironment():

    def __init__(self, record='data'):
        self.record = record
        self.reset()

    def set_players(self, black, white):
        self.players = {pachi_py.BLACK: black,
                        pachi_py.WHITE: white}

        self.state_hist = [None, self.state]
        self.action_hist = [None]

    def run(self):

        if self.record is not None:
            self.game_hist = {pachi_py.BLACK: [],
                              pachi_py.WHITE: []}

        stop = False
        while not stop:
            action = self.players[self.current_player].act(
                self.state, self.state_hist[-2], self.action_hist[-1])
            self.state = self.state.act(action)
            self.state_hist.append(self.state)
            self.action_hist.append(action)
            self.current_player = self.state.color
            stop = self.state.board.is_terminal

            if self.record is not None:
                board = roll_axis(self.state_hist[-1].board.encode())
                self.game_hist[self.current_player].append((board, action))

        print('INFO: Game is terminated. Score: {}'.format(self.state.board.official_score))

        if self.record is not None:
            game = {'hist': self.game_hist, 'result': self.state.board.official_score}
            joblib.dump(game, os.path.join(self.record, str(int(time.time())) + '.dat'))

    def reset(self):
        self.state = GoState(pachi_py.CreateBoard(9), pachi_py.BLACK)
        self.current_player = pachi_py.BLACK
        self.players = None
        os.makedirs(self.record, exist_ok=True)


# ---- Pachi Policy
def make_pachi_policy(board, engine_type='uct', threads=1, pachi_timestr=''):
    engine = pachi_py.PyPachiEngine(board, engine_type, six.b('threads=%d' % threads))

    def pachi_policy(curr_state, prev_state, prev_action):
        if prev_state is not None:
            assert engine.curr_board == prev_state.board, 'Engine internal board is inconsistent with provided board. The Pachi engine must be called consistently as the game progresses.'
            prev_coord = _action_to_coord(prev_state.board, prev_action)
            engine.notify(prev_coord, prev_state.color)
            engine.curr_board.play_inplace(prev_coord, prev_state.color)
        out_coord = engine.genmove(curr_state.color, pachi_timestr)
        out_action = _coord_to_action(curr_state.board, out_coord)
        engine.curr_board.play_inplace(out_coord, curr_state.color)
        return out_action
    return pachi_policy


# --- GYM envs

def _coord_to_action(board, c):
    '''Converts Pachi coordinates to actions'''
    if c == pachi_py.PASS_COORD:
        return board.size**2  # pass
    if c == pachi_py.RESIGN_COORD:
        return board.size**2 + 1  # resign
    i, j = board.coord_to_ij(c)
    return i * board.size + j


def _action_to_coord(board, a):
    '''Converts actions to Pachi coordinates'''
    if a == board.size**2:
        return pachi_py.PASS_COORD
    if a == board.size**2 + 1:
        return pachi_py.RESIGN_COORD
    return board.ij_to_coord(a // board.size, a % board.size)


def str_to_action(board, s):
    return _coord_to_action(board, board.str_to_coord(s))


class GoState(object):
    '''
    Go game state. Consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is different
    from Pachi's internal "coord_t" encoding.
    '''

    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in [pachi_py.BLACK, pachi_py.WHITE], 'Invalid player color'
        self.board, self.color = board, color

    def act(self, action):
        '''
        Executes an action for the current player
        Returns:
            a new GoState with the new board and the player switched
        '''
        return GoState(
            self.board.play(_action_to_coord(self.board, action), self.color),
            pachi_py.stone_other(self.color))

    def __repr__(self):
        return 'To play: {}\n{}'.format(pachi_py.color_to_str(self.color), repr(self.board))

# --- misc


def roll_axis(ob):
    """ Change observations from CxHxW to HxWxC"""
    ob = np.swapaxes(ob, 0, 1)
    ob = np.swapaxes(ob, 1, 2)
    return ob


def test():
    black = Player(pachi_py.BLACK)
    white = Player(pachi_py.WHITE)
    env = TrainingEnvironment(black, white)
    env.run()
    return env.state


if __name__ == "__main__":
    print(test())
