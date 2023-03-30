import tensorflow as tf
import os
import numpy as np
from strategy.state_action_qnet import StateActionQNet
from utils.model_mode import ModelMode
from logger import logger


class QNetWrapper:
    def __init__(self, 
                 state_action_size, 
                 net_args=dict(),
                 lr=.01,
                 max_grad=1,
                 ckpt_dir='ckpts/qnet',
                 model_name='q-net',
                 session_config=None):
        graph = tf.Graph()
        with graph.as_default():
            with tf.variable_scope('pred_qnet'):
                self.pred_qnet = StateActionQNet(state_action_size, mode=ModelMode.pred, **net_args)
            
            with tf.variable_scope('train_qnet'):
                train_qnet = StateActionQNet(state_action_size, **net_args)
            
            self.copy_var_ops = copy_vars('train_qnet', 'pred_qnet')

            loss = train_qnet.loss
            self.step = tf.Variable(initial_value=0, 
                                    name='global_step', 
                                    trainable=False, 
                                    collections=[tf.GraphKeys.GLOBAL_STEP, 
                                                tf.GraphKeys.GLOBAL_VARIABLES])
            self.learning_rate = tf.Variable(lr, trainable=False, name='lr')
            with tf.variable_scope('optimzer', reuse=tf.AUTO_REUSE):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                if max_grad > 0:
                    grads = optimizer.compute_gradients(loss)
                    for i, (g, v) in enumerate(grads):
                        if g is not None:
                            grads[i] = (tf.clip_by_norm(g, max_grad), v)
                    self.train_op = optimizer.apply_gradients(grads, global_step=self.step)
                else:
                    self.train_op = optimizer.minimize(loss, global_step=self.step)

            self.loss_summary = tf.summary.scalar('loss', loss)
            self.merged_summary = tf.summary.merge_all()
            self.train_qnet = train_qnet

            self.ckpt_dir = ckpt_dir
            self.model_name = model_name
            self.saver = tf.train.Saver()

            # Initialize session.
            self.sess = tf.Session(graph=graph, config=session_config)
            with self.sess.as_default():
                sess = self.sess
                self.summary_writer = tf.summary.FileWriter('train/qnet', sess.graph)

                # Initialize variables.
                meta_path = os.path.join(self.ckpt_dir, f'{self.model_name}.meta')
                if os.path.exists(meta_path):
                    saver = tf.train.import_meta_graph(meta_path)
                    saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_dir))
                else:
                    sess.run(tf.global_variables_initializer())
                    sess.run(self.copy_var_ops)
            
    def train(self, dataset, num_epoch=1, save_every=1):
        qnet = self.train_qnet
        with self.sess.as_default():
            sess = self.sess
            for i in range(1, num_epoch + 1):
                for state_actions, expected_values in dataset:
                    _, step, value, loss, summary = sess.run(
                        [self.train_op, self.step, qnet.value, qnet.loss, self.merged_summary],
                        feed_dict={
                            qnet.state_action: state_actions,
                            qnet.expected_value: expected_values,
                        }
                    )
                    self.summary_writer.add_summary(summary)
                    logger.debug(f'Epoch: {i}, step: {step}, value[0]: {value[0]}, '
                                 f'expected value[0]: {expected_values[0]}, loss: {loss}.')
                    logger.debug('actual-values vs expected-values')
                    for va, ve in zip(value.tolist(), expected_values.tolist()):
                        logger.debug(f'{va} : {ve}')

                if i % save_every == 0:
                    self.saver.save(sess, os.path.join(self.ckpt_dir, self.model_name))
                    logger.debug(f'Checkpoint saved to {self.ckpt_dir}.')
            # Update prediction q-network after training.
            sess.run(self.copy_var_ops)
    
    def predict(self, state_actions, batch_size=4):
        values = []
        n_data = state_actions.shape[0]
        with self.sess.as_default():
            sess = self.sess
            for i in range(0, n_data, batch_size):
                state_action_batch = state_actions[i:(i + batch_size), :]
                value_batch = sess.run(self.pred_qnet.value, 
                                       feed_dict={self.pred_qnet.state_action: state_action_batch})
                values.append(value_batch)
        values = np.concatenate(values, axis=0)
        return values
  
def copy_vars(from_scope, to_scope):
    var_dict = {}
    for var in tf.get_collection(tf.GraphKeys.VARIABLES, from_scope):
        var_name = var.name[len(from_scope):]
        var_dict[var_name] = var
    ops = []
    for var in tf.get_collection(tf.GraphKeys.VARIABLES, to_scope):
        var_name = var.name[len(to_scope):]
        if var_name in var_dict:
            ops.append(tf.assign(var, var_dict[var_name]))
        else:
            raise RuntimeError(f'Var {var_name} not found in scope {from_scope}.')
    return ops
