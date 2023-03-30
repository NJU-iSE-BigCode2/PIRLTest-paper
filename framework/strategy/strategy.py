import random
import numpy as np
from strategy.qnet_wrapper import QNetWrapper
from logger import logger


class EpsilonGreedyStrategy:
    def __init__(self, 
                 qnet_wrapper_args, 
                 batch_size=4, 
                 action_type_coefficients=None,
                 epsilon=.1,
                 num_warm_ups=4):
        self.qnet_wrapper = QNetWrapper(**qnet_wrapper_args)
        self.batch_size = batch_size
        self.action_type_coefficients = action_type_coefficients
        self.epsilon = epsilon
        self.warm_up_countdown = num_warm_ups
    
    def train_qnet(self, train_data, num_epoch=1):
        self.qnet_wrapper.train(train_data, num_epoch=num_epoch, save_every=num_epoch)

    def warm_up(self, state_embedding, actions, action_embeddings):
        k = random.randint(0, len(actions) - 1)
        action = actions[k]
        embedding = np.concatenate([state_embedding, action_embeddings[k]])
        self.warm_up_countdown -= 1
        return action, embedding

    def explore(self, state_embedding, actions, action_embeddings):
        if self.warm_up_countdown > 0:
            return self.warm_up(state_embedding, actions, action_embeddings)

        num_actions = len(actions)
        if random.random() < self.epsilon:
            idx = random.randint(0, num_actions - 1)
            return actions[idx], np.concatenate([state_embedding, action_embeddings[idx]])
        state_embeddings = np.tile(state_embedding, (num_actions, 1))
        state_actions = np.concatenate([state_embeddings, action_embeddings], axis=1)
        scores = self.qnet_wrapper.predict(state_actions, batch_size=self.batch_size)
        logger.debug('action: score')
        for a, s in zip(actions, scores.tolist()):
            logger.debug(f'{a}: {s}')

        # To prevent negative scores, we substract them by their minimum.
        scores -= scores.min()
        if self.action_type_coefficients is not None:
            action_weights = np.array(self.action_type_coefficients)[[a[0] for a in actions]]
            scores *= action_weights
        best_idx = np.argmax(scores)
        best_action = actions[best_idx]
        best_sa_embedding = state_actions[best_idx]
        return best_action, best_sa_embedding

class BoltzmannStrategy:
    def __init__(self, 
                 qnet_wrapper_args,
                 batch_size=4,
                 action_type_coefficients=None):
        self.qnet_wrapper = QNetWrapper(**qnet_wrapper_args)
        self.batch_size = batch_size
        self.action_type_coefficients = action_type_coefficients

    def train_qnet(self, train_data, num_epoch=1):
        self.qnet_wrapper.train(train_data, num_epoch=num_epoch, save_every=num_epoch)

    def explore(self, state_embedding, actions, action_embeddings):
        num_actions = len(actions)
        state_embeddings = np.tile(state_embedding, (num_actions, 1))
        state_actions = np.concatenate([state_embeddings, action_embeddings], axis=1)
        scores = qnet_wrapper.predict(state_actions, batch_size=batch_size)
        logger.debug('action: score')
        for a, s in zip(actions, scores.tolist()):
            logger.debug(f'{a}: {s}')

        scores -= scores.min()
        action_weights = np.array(self.action_type_coefficients)[[a[0] for a in actions]]
        if action_weights is not None:
            scores *= action_weights

        exps = np.exp(scores)
        probs = exps / np.sum(exps)
        r = random.random()
        k = 0
        for i, p in enumerate(probs):
            k += p
            if r < k:
                return actions[i], action_embeddings[i]
        logger.warn(f'Random number >= 1 not supposed to happen: {r}')
        return actions[-1], action_embeddings[-1]
