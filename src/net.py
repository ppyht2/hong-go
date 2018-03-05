import os
import time
import tensorflow as tf
from tensorflow.python.layers.core import flatten, dense
from tensorflow.python.layers.convolutional import conv2d
from tensorflow.python.layers.normalization import batch_normalization
N_ACTION = 83


class DualRes():
    def __init__(self, n_feature, n_filter, n_residual, game_size):
        self.n_feature = n_feature
        self.n_filter = n_filter
        self.n_residual = n_residual
        self.game_size = game_size
        self.valid = None
        self.sess_id = time.strftime('%Y-%m-%dT%H:%M:%SZ')

    def build_graph(self):

        def residual_block(inputs):
            c1 = conv2d(inputs, filters=self.n_filter, kernel_size=3, padding='SAME')
            b1 = batch_normalization(c1)
            h1 = tf.nn.relu(b1)
            c2 = conv2d(h1, filters=self.n_filter, kernel_size=3, padding='SAME')
            b2 = batch_normalization(c2)
            s1 = b2 + c1
            h2 = tf.nn.relu(s1)
            return h2

        input_features = tf.placeholder(tf.float32,
                                        [None, self.game_size, self.game_size, self.n_feature])
        input_player = tf.placeholder(tf.float32, [None])
        # Training
        train_values = tf.placeholder(tf.float32, [None])
        train_pi = tf.placeholder(tf.int32, [None])

        with tf.variable_scope('player_channel'):
            pc = tf.expand_dims(input_player, -1)
            pc = pc * tf.ones([1, self.game_size], dtype=tf.float32)
            pc = tf.expand_dims(pc, -1)
            pc = pc * tf.ones([1, self.game_size, self.game_size], dtype=tf.float32)
            pc = tf.expand_dims(pc, -1)

        inputs = tf.concat((input_features, pc), 3)

        with tf.variable_scope('convolutional_block'):
            conv_layer = conv2d(inputs, filters=self.n_filter,
                                kernel_size=3, padding='SAME')
            batch_norm_layer = batch_normalization(
                conv_layer)  # mini-go uses a different config
            h1 = tf.nn.relu(batch_norm_layer)

        with tf.variable_scope('residual_blocks'):
            current_layer = h1
            for i in range(self.n_residual):
                current_layer = residual_block(current_layer)

        with tf.variable_scope('policy_head'):
            policy_conv = conv2d(current_layer, filters=2, kernel_size=1)
            policy_batch = tf.nn.relu(batch_normalization(policy_conv))
            policy_flat = flatten(policy_batch)
            policy_logits = dense(policy_flat, N_ACTION)

        with tf.variable_scope('value_head'):
            value_conv = conv2d(current_layer, filters=1, kernel_size=1)
            value_batch = tf.nn.relu(batch_normalization(value_conv))
            value_flat = flatten(value_batch)
            value_dense = dense(value_flat, 256, activation=tf.nn.relu)
            value_output = tf.nn.tanh(dense(value_dense, 1))

        policy_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=train_pi)
        policy_cost = tf.reduce_mean(policy_xent)
        value_cost = tf.reduce_mean(tf.square(value_output - train_values))
        combined_cost = policy_cost + value_cost

        global_step = tf.Variable(0, trainable=False)
        optimiser = tf.train.MomentumOptimizer(1e-3, 0.9)
        train_op = optimiser.minimize(combined_cost, global_step=global_step)

        # tensorboard
        tf.summary.scalar('policy_cost', policy_cost)
        tf.summary.scalar('value_cost', value_cost)
        tf.summary.scalar('total_cost', combined_cost)
        merged_summary = tf.summary.merge_all()
        self.summary = merged_summary

        #
        self.global_step = global_step

        self.input_features = input_features
        self.input_player = input_player

        self.train_values = train_values
        self.train_pi = train_pi

        self.policy_logits = policy_logits
        self.value_output = value_output

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.train_op = train_op

        self.writer = tf.summary.FileWriter(os.path.join('tb', self.sess_id))
        self.writer.add_graph(self.sess.graph)

        # debug
        self.inputs = inputs
        self.policy_cost = policy_cost
        self.policy_xent = policy_xent

    def train(self, s, a, v, p):
        d = {self.input_features: s,
             self.train_values: v,
             self.train_pi: a,
             self.input_player: p,
             }

        gs = self.sess.run(self.global_step)
        if gs % 50 == 0:
            s = self.sess.run(self.summary, d)
            self.writer.add_summary(s, gs)
            self.writer.flush()

        self.sess.run(self.train_op, d)

    def add_validation(self, s, a, v, p):
        self.valid = {self.input_features: s,
                      self.train_values: v,
                      self.train_pi: a,
                      self.input_player: p,
                      }

    def eval(self, s, p):
        pi, v = self.sess.run([self.policy_logits, self.value_output],
                              feed_dict={self.input_features: inputs})
        return pi, v

    def save(self, path):
        pass

    def load(self, path):
        pass


if __name__ == "__main__":
    print('Test')
