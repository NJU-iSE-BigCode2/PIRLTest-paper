import tensorflow as tf
from img_process.layout.tree_dist_net import TreeEditDistNet
from img_process.layout.tree_dist_dataset import TreePairDataset
import logging
import os
import glob


class TreeEditDistTrainer:
    def __init__(self, 
                 lr=.01, 
                 lr_decay_rate=.95,
                 lr_decay_steps=10000,
                 max_grad=1,
                 ckpt_dir='ckpts/ted',
                 model_name='tree_edit_dist',
                 net_args={}):
        ted_model = TreeEditDistNet(**net_args)
        loss = ted_model.loss
        self.step = tf.Variable(initial_value=0, 
                                name='global_step', 
                                trainable=False, 
                                collections=[tf.GraphKeys.GLOBAL_STEP, 
                                             tf.GraphKeys.GLOBAL_VARIABLES])
        self.learning_rate = tf.train.exponential_decay(lr, self.step, lr_decay_steps, lr_decay_rate, False, 'lr')
        #self.learning_rate = tf.Variable(lr, trainable=False, name='lr')
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            if max_grad > 0:
                grads = optimizer.compute_gradients(loss)
                for i, (g, v) in enumerate(grads):
                    if g is not None:
                        grads[i] = (tf.clip_by_norm(g, max_grad), v)  # clip grad to 5
                self.train_op = optimizer.apply_gradients(grads, global_step=self.step)
            else:
                self.train_op = optimizer.minimize(loss, global_step=self.step)
            
        self.loss_summary = tf.summary.scalar('loss', loss)
        self.merged_summary = tf.summary.merge_all()
        self.ted_model = ted_model
        self.ckpt_dir = ckpt_dir
        self.model_name = model_name
        self.saver = tf.train.Saver()

    def train(self, dataset, epochs=1000, save_every=1000):
        '''
        :param dataset: training dataset
        :type dataset: tree_dist_dataset.TreePairDataset
        :param epochs: number of epochs
        :param save_every: save every ? steps
        '''
        model = self.ted_model
        saver = self.saver

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('train/ted', sess.graph)
            self.init_vars(sess)

            j = 0
            for seq_a, len_a, seq_b, len_b, true_dis in dataset:
                target_tensors = [self.train_op, 
                                  self.step, 
                                  model.distance,
                                  model.loss, 
                                  self.merged_summary]
                _, step, dis, train_loss, summary = sess.run(target_tensors, feed_dict={
                    model.seq_a: seq_a,
                    model.len_a: len_a,
                    model.seq_b: seq_b,
                    model.len_b: len_b,
                    model.true_dis: true_dis,
                })
                j += 1
                writer.add_summary(summary, j)
                logging.info(f'run {j}, global_step: {step}, '
                             f'pred dis: {dis}, true dis: {true_dis}, loss: {train_loss}.')
                if j % save_every == 0:
                    saver.save(sess, os.path.join(self.ckpt_dir, self.model_name))

    def init_vars(self, sess):
        meta_path = os.path.join(self.ckpt_dir, f'{self.model_name}.meta')
        if os.path.exists(meta_path):
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_dir))
            var_names = [v.name for v in tf.all_variables()]
            for name in var_names:
                print(name)
        else:
            sess.run(tf.global_variables_initializer())

def main():
    import sys

    logging.basicConfig(level=logging.DEBUG)

    dataset = TreePairDataset()
    trainer = TreeEditDistTrainer()
    save_every = 100
    logging.info('Default save the model args every 100 runs.')
    trainer.train(dataset, save_every=save_every)


if __name__ == '__main__':
    main()
