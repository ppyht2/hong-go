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
    train_size = 34000
    train_epoch = train_size * 2
    batch_names = [os.path.join(batch_dir, i) for i in os.listdir(batch_dir)]
    create_batch(max(0, train_size - len(batch_names)))
    batch_names = [os.path.join(batch_dir, i) for i in os.listdir(batch_dir)]

    # Let's load up some random batch as the validation sets
    test_size = 32
    test_names = np.random.choice(batch_names, test_size)
    test_data = []
    for n in test_names:
        batch = joblib.load(n)
        s, a, v, p = process_batch(batch)
        test_data.append((s, a, v, p))
    test_data = zip(*test_data)
    test_data = [np.concatenate(f, axis=0) for f in test_data]

    # Model
    model = DualRes(n_feature=3, n_filter=128, n_residual=12, game_size=9)
    model.build_graph()
    model.add_validation(*test_data)

    batch_idx = 0

    # Training
    for e in range(train_epoch):
        batch = joblib.load(batch_names[batch_idx])
        s, a, v, p = process_batch(batch)
        model.train(s, a, v, p)
        batch_idx += 1
        if batch_idx < train_size:
            batch_idx = 0
            np.random.shuffle(batch_names)

    model.save('v1')
    model.load('v1')


if __name__ == "__main__":
    main()
