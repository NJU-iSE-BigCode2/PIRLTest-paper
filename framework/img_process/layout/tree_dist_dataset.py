import random
from img_process.layout.layout_extraction import generate_layout_seqs
from img_process.layout.layout_utils import tree_distance, cvt_tree_to_int_seq, count_tree_nodes


def generate_tree(max_num_nodes=300, min_num_nodes=1, max_num_child_per_node=10):
    num_nodes = random.randint(min_num_nodes, max_num_nodes)
    queue = []
    root = [1, []]
    queue.append(root[1])
    countdown = num_nodes - 1
    while countdown > 0:
        children = queue.pop(0)
        num_children = random.randint(1, max_num_child_per_node)
        for _ in range(num_children):
            child = [1, []]
            children.append(child)
            queue.append(child[1])
        countdown -= num_children
    return root

class TreePairDataset:
    def __init__(self, buffer_size=64):
        self.buffer_size = buffer_size
        self.buffer = []
        self.p = 0
        self._reset()
    
    def _reset(self):
        self.buffer = [generate_tree() for _ in range(self.buffer_size << 1)]
        self.p = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.p >= self.buffer_size:
            self._reset()
        i = self.p
        j = self.p + self.buffer_size
        self.p += 1
        tree_a = self.buffer[i]
        seq_a = cvt_tree_to_int_seq(tree_a)
        len_a = count_tree_nodes(tree_a)
        tree_b = self.buffer[j]
        seq_b = cvt_tree_to_int_seq(tree_b)
        len_b = count_tree_nodes(tree_b)
        true_dis = tree_distance(tree_a, tree_b)
        if true_dis == 0:
            return self.__next__()
        return seq_a, len_a, seq_b, len_b, true_dis

