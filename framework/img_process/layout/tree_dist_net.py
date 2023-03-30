import tensorflow as tf
from img_process.layout.tree_embedding import TreeEmbeddingNet
from utils.model_mode import ModelMode


class TreeEditDistNet:
    def __init__(self, **kwargs):
        if 'mode' not in kwargs:
            self.mode = ModelMode.train
        else:
            self.mode = kwargs['mode']
        
        self.seq_a = tf.placeholder(dtype=tf.int32, shape=[None])
        self.len_a = tf.placeholder(dtype=tf.int32, shape=[])
        self.seq_b = tf.placeholder(dtype=tf.int32, shape=[None])
        self.len_b = tf.placeholder(dtype=tf.int32, shape=[])
        if not self.mode == ModelMode.pred:
            self.true_dis = tf.placeholder(dtype=tf.float32, shape=[])
        
        with tf.variable_scope('tree_edit_dist') as scope:
            embedding_model_a = TreeEmbeddingNet(**kwargs)
            embedding_model_a.input_seq = self.seq_a
            self.embedding_a = embedding_model_a.build()

            scope.reuse_variables()

            embedding_model_b = TreeEmbeddingNet(**kwargs)
            embedding_model_b.input_seq = self.seq_b
            self.embedding_b = embedding_model_b.build()
        
        if self.mode == ModelMode.train:
            with tf.name_scope('square_loss'):
                # To avoid sqrt(0) -> grad = inf -> nan problem.
                diff = tf.reduce_sum(tf.square(self.embedding_a - self.embedding_b))
                eps = 1e-8
                self.distance = tf.sqrt(tf.maximum(diff, eps))
                max_nodes = tf.cast(tf.maximum(self.len_a, self.len_b), dtype=tf.float32)
                self.loss = tf.square((self.distance - self.true_dis) / max_nodes)
    
    @property
    def outputs(self):
        if self.mode == ModelMode.train:
            return self.distance, self.loss
        else:
            return self.distance
    
    @property
    def embeddings(self):
        return self.embedding_a, self.embedding_b

