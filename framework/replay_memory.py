import random
import os
import numpy as np
import sys
from rewards import generate_reward


class ReplayMemory:
    def __init__(self, ptm, size_limit=1 << 30, dump_dir='replay-memory', gamma=.8):
        self.ptm = ptm  # PageTransactionManager
        self.memory = []
        self.size_limit = size_limit
        self.current_size = 0
        self.gamma = gamma

    @staticmethod
    def _get_item_size(item):
        old_sa, pt, new_s, new_a = item
        return sys.getsizeof(old_sa) + sys.getsizeof(pt) + sys.getsizeof(new_s) + sys.getsizeof(new_a)

    def append(self, state_action, page_transition, new_state, new_action_embeddings):
        item = (state_action, page_transition, new_state, new_action_embeddings)
        item_size = ReplayMemory._get_item_size(item)
        assert item_size <= self.size_limit, f'Size limit smaller than a single item: {self.size_limit} < {item_size}'
        while self.current_size + item_size > self.size_limit:
            self.current_size -= ReplayMemory._get_item_size(self.memory.pop(0))
        self.memory.append(item)
        self.current_size += item_size

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, key):
        '''
        :param key: An index or several indices.
        :type key: Int or int-list.
        '''
        if type(key) == int:
            return self.memory[key]
        if type(key) == list:
            return [self.memory[i] for i in key]
        raise RuntimeError(f'Invalid key: {key}.')

    def _select_train_ids(self, n):
        k = len(self)
        ids = list(range(k))
        if n >= k:
            random.shuffle(ids)
            return ids

        # Select both recent items and other random items.
        # This is to make it learn more from recent transitions.
        num_recents = n // 2
        recents = ids[k - num_recents:]
        remainings = ids[:k - num_recents]
        random.shuffle(remainings)
        num_rand = n - num_recents
        rands = remainings[:num_rand]
        results = recents + rands
        random.shuffle(results)
        return results
    
    def get_train_data(self, qnet_wrapper, pred_kwargs=dict(), n=16, batch_size=4):
        selected_ids = self._select_train_ids(n)
        data = self[selected_ids]

        # Compute expected q-values.
        expected_qs = []
        for d in data:
            new_action_embeddings = d[3]
            num_new_actions = new_action_embeddings.shape[0]
            new_state_embeddings = np.tile(d[2], (num_new_actions, 1))
            new_state_actions = np.concatenate([new_state_embeddings, new_action_embeddings], axis=1)
            # Check invalid arguments.
            assert not np.any(np.isnan(new_action_embeddings)), 'Ndarray new_action_embeddings has NaN values.'
            assert not np.any(np.isnan(d[2])), 'Ndarray new_state_embedding has NaN values.'
            scores = qnet_wrapper.predict(new_state_actions, **pred_kwargs)
            assert not np.any(np.isnan(scores)), 'Ndarray scores has NaN values.'
            # As exploration goes on, the explore rate and transition count could change, 
            # such that the reward computed before becomes out-of-date. So we have to
            # compute the reward here instead of after the execution of the action.
            reward = generate_reward(*d[1], self.ptm)
            expected_qs.append(reward + self.gamma * np.max(scores))

        # Make batches.
        batches = []
        for i in range(0, min(n, len(data)), batch_size):
            state_actions = np.stack([d[0] for d in data[i:(i + batch_size)]])
            expected_values = np.array(expected_qs[i:(i + batch_size)])
            batches.append((state_actions, expected_values))
        return batches

