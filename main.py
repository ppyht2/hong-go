import os
import pachi_py
import joblib
import numpy as np
from src.net import DualRes
from src.game import TrainingEnvironment, PachiPlayer


def process_batch(batch):
    def process_colour(c):
        states, actions = zip(*batch['hist'][c])
        states = np.asanyarray(states)
        actions = np.asanyarray(actions)

        player = np.ones_like(actions) * (c - 1)
        value = np.ones_like(actions) * np.sign(batch['result']) * (c * 2 - 3)
        return (states, actions, value, player)

    full = [process_colour(i) for i in (pachi_py.BLACK, pachi_py.WHITE)]
    full = zip(*full)
    full = [np.concatenate(f, axis=0) for f in full]
    return full


def create_batch(n):
    for i in range(n):
        env = TrainingEnvironment()
        black_player = PachiPlayer(pachi_py.BLACK, env.state.board)
        white_player = PachiPlayer(pachi_py.WHITE, env.state.board)
        env.set_players(black_player, white_player)
        env.run()


def main():
    batch_dir = 'data'
    train_size = 20000
    batch_names = [os.path.join(batch_dir, i) for i in os.listdir(batch_dir)]
    create_batch(max(1, train_size - len(batch_names)))
    batch_names = [os.path.join(batch_dir, i) for i in os.listdir(batch_dir)]
    model = DualRes(n_feature=3, n_filter=128, n_residual=6, game_size=9)
    model.build_graph()

    batch_idx = 0

    for e in range(train_size):
        batch = joblib.load(batch_names[batch_idx])
        s, a, v, p = process_batch(batch)
        model.train(s, a, v, p)
        batch_idx += 1


if __name__ == "__main__":
    main()
