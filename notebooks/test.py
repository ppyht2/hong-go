import gym
import numpy as np
import pachi_py
from gym.envs.board_game.go import _coord_to_action, GoState, _action_to_coord
import six
ENV_ID = 'Go9x9-v0'
# Action 82 is resign


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


def sim_game(player):
    assert player in (pachi_py.BLACK, pachi_py.WHITE)
    enemy = pachi_py.stone_other(player)

    ob = []
    a = []

    state = GoState(pachi_py.CreateBoard(9), pachi_py.BLACK)
    player_policy = make_pachi_policy(board=state.board.clone(
    ), engine_type=six.b('uct'), pachi_timestr=six.b('_2400'))

    last_enemy_action = None
    last_state = None

    while not state.board.is_terminal:
        if state.color == player:
            print(last_state)
            player_action = player_policy(state, last_state, last_enemy_action)
            state = state.act(player_action)
            assert state.color != player
            continue
        elif state.color == enemy:
            legal_coord = state.board.get_legal_coords(enemy)
            legal_actions = np.array([_coord_to_action(state.board, c) for c in legal_coord])
            last_enemy_action = np.random.choice(legal_actions)
            last_state = state
            state = state.act(last_enemy_action)
            continue
        else:
            raise NotImplementedError

    return state


ob = sim_game(pachi_py.WHITE)
