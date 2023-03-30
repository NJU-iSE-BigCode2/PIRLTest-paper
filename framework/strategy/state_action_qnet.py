from utils.model_mode import ModelMode
import tensorflow as tf


class StateActionQNet:
    def __init__(self, 
                 state_action_size,
                 hidden_size=512, 
                 num_layers=4, 
                 activation_fn=tf.nn.leaky_relu,
                 mode=ModelMode.train,
                 max_value=5):
        state_action = tf.placeholder(dtype=tf.float32, shape=[None, state_action_size])
        expected_value = tf.placeholder(dtype=tf.float32, shape=[None])
        hidden_layer = state_action
        for i in range(num_layers):
            with tf.variable_scope(f'hidden_{i}'):
                hidden_layer = tf.layers.dense(hidden_layer, hidden_size, activation=activation_fn)
        value = tf.layers.dense(hidden_layer, 1, activation=None)
        #value = tf.sigmoid(raw_value) * max_value
        value = tf.squeeze(value, axis=1)

        if mode == ModelMode.train:
            batch_size = tf.cast(tf.shape(state_action)[0], dtype=tf.float32)
            self.loss = tf.reduce_sum(tf.square(expected_value - value)) / batch_size

        self.state_action = state_action
        self.expected_value = expected_value
        self.value = value
