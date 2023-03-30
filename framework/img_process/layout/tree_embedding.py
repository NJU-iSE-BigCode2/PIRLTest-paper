import tensorflow as tf
from utils.model_mode import ModelMode


class TreeEmbeddingNet:
    def __init__(self,
                 num_lstm_units=512,
                 lstm_dropout_keep_prob=.8,
                 hidden_size=512,
                 mode=ModelMode.train,
                 vocab_size=4,
                 embed_size=512,):
        self.num_lstm_units = num_lstm_units
        self.lstm_dropout_keep_prob = lstm_dropout_keep_prob
        self.hidden_size = hidden_size
        self.mode = mode
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Input
        self.input_seq = None
        # Output
        self.embedding = None
    
    def build(self):
        with tf.variable_scope('tree_embedding'):
            with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
                embedding_dict = tf.get_variable('embedding_dict', [self.vocab_size, self.embed_size])
                embeddings = tf.nn.embedding_lookup(embedding_dict, self.input_seq)
                embeddings = tf.expand_dims(embeddings, axis=0)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.num_lstm_units)
            if self.mode == ModelMode.train:
                lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, 
                                                          input_keep_prob=self.lstm_dropout_keep_prob,
                                                          output_keep_prob=self.lstm_dropout_keep_prob)
            with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE) as lstm_scope:
                zero_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
                _, initial_state = lstm_cell(embeddings[:, 0, :], zero_state)
                _, final_state = tf.nn.dynamic_rnn(lstm_cell, 
                                                   embeddings, 
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope=lstm_scope)
            final_state = tf.concat(final_state, axis=1)
            with tf.name_scope('dfs_fc'):
                output = tf.layers.dense(final_state, self.hidden_size, activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE)
        self.embedding = output
        return output

    def build_input(self):
        self.input_seq = tf.placeholder(dtype=tf.int32, shape=[None])
    
