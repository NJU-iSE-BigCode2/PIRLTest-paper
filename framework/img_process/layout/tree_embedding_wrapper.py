import os
import tensorflow as tf
from img_process.layout.tree_embedding import TreeEmbeddingNet


class TreeEmbeddingWrapper:
    def __init__(self,
                 ckpt_dir='ckpts/ted',
                 var_scope='tree_edit_dist',
                 model_kwargs=dict(),
                 session_config=None):
        g = tf.Graph()
        with g.as_default():
            model = TreeEmbeddingNet(**model_kwargs)
            model.build_input()
            model.build()
            tree_embedding_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='tree_embedding')
            var_map = {f'{var_scope}/{v.name[:-2]}': v for v in tree_embedding_vars}
            saver = tf.train.Saver(var_map)

            sess = tf.Session(graph=g, config=session_config)
            # Restore vars.
            with sess.as_default():
                saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

            self.sess = sess
            self.model = model
    
    def predict(self, input_seq):
        with self.sess.as_default():
            embedding = self.sess.run(self.model.embedding, feed_dict={
                self.model.input_seq: input_seq,
            })
        return embedding


tew = TreeEmbeddingWrapper(session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)))

