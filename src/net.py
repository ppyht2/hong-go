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
            policy_softmax = tf.nn.softmax(policy_flat)
            policy_pred = tf.argmax(policy_logits, axis=1)
            _, accuracy = tf.metrics.accuracy(labels=train_pi, predictions=policy_pred)
            tf.summary.scalar('policy_accuracy', accuracy)

        with tf.variable_scope('value_head'):
            value_conv = conv2d(current_layer, filters=1, kernel_size=1)
            value_batch = tf.nn.relu(batch_normalization(value_conv))
            value_flat = flatten(value_batch)
            value_dense = dense(value_flat, 256, activation=tf.nn.relu)
            value_output = tf.nn.tanh(dense(value_dense, 1))

        policy_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=train_pi)
        policy_cost = tf.reduce_mean(policy_xent)
        value_err = tf.square(tf.squeeze(value_output) - train_values)
        value_cost = tf.reduce_mean(value_err)

        l2_cost = 1e-4 * tf.add_n([tf.nn.l2_loss(v)
                                   for v in tf.trainable_variables() if 'bias' not in v.name])
        combined_cost = policy_cost + 2.0 * value_cost + l2_cost

        global_step = tf.Variable(0, trainable=False)
        optimiser = tf.train.MomentumOptimizer(1e-3, 0.9)
        train_op = optimiser.minimize(combined_cost, global_step=global_step)

        # tensorboard
        tf.summary.scalar('policy_cost', policy_cost)
        tf.summary.scalar('value_cost', value_cost)
        tf.summary.scalar('l2_cost', l2_cost)
        tf.summary.scalar('total_cost', combined_cost)
        merged_summary = tf.summary.merge_all()
        self.summary = merged_summary

        #
        self.value_cost = value_cost
        self.value_err = value_err

        self.global_step = global_step

        self.input_features = input_features
        self.input_player = input_player

        self.train_values = train_values
        self.train_pi = train_pi

        self.policy_logits = policy_logits
        self.value_output = value_output

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.train_op = train_op

        self.train_writer = tf.summary.FileWriter(os.path.join('tb', self.sess_id, 'TRAIN'))
        self.valid_writer = tf.summary.FileWriter(os.path.join('tb', self.sess_id, 'VALID'))
        self.train_writer.add_graph(self.sess.graph)

        # debug
        self.inputs = inputs
        self.policy_cost = policy_cost
        self.policy_xent = policy_xent
        self.policy_softmax = policy_softmax

    def train(self, s, a, v, p):
        d = {self.input_features: s,
             self.train_values: v,
             self.train_pi: a,
             self.input_player: p,
             }

        # Tensorboard
        gs = self.sess.run(self.global_step)
        if gs % 50 == 0:
            self.write_summary(gs, d)

        # Training
        self.sess.run(self.train_op, d)

    def write_summary(self, global_step, train_dict):
        # Training summary
        self.sess.run(tf.local_variables_initializer())
        s = self.sess.run(self.summary, train_dict)
        self.train_writer.add_summary(s, global_step)
        self.train_writer.flush()
        # Validation summary
        assert self.valid is not None
        self.sess.run(tf.local_variables_initializer())
        s = self.sess.run(self.summary, self.valid)
        self.valid_writer.add_summary(s, global_step)
        self.valid_writer.flush()

    def add_validation(self, s, a, v, p):
        self.valid = {self.input_features: s,
                      self.train_values: v,
                      self.train_pi: a,
                      self.input_player: p,
                      }

    def eval(self, s, p):
        d = {self.input_features: s, self.input_player: p}
        pi, v = self.sess.run([self.policy_logits, self.value_output], d)
        return pi, v

    def save(self, version):
        saver = tf.train.Saver()
        save_path = os.path.join('models', version)
        os.makedirs(save_path, exist_ok=True)
        saver.save(self.sess, os.path.join(save_path, 'v1'))

    def load(self, version):
        assert False
        # This function does not work
        # Clear out existing sessions and graphs
        tf.reset_default_graph()
        self.sess = tf.Session()
        # load from graphs and sessions
        meta = os.path.join('models', version, version + '.meta')
        saver = tf.train.import_meta_graph(meta)
        saver.restore(self.sess, tf.train.latest_checkpoint(os.path.join('models', version)))

        # Carry on here


if __name__ == "__main__":
    print('Test')
