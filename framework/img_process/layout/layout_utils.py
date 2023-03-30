from apted import APTED, helpers


def count_tree_nodes(tree):
    result = 0
    for child in tree[1]:
        result += count_tree_nodes(child)
    result += 1
    return result
    
def cvt_tree_to_bracket_seq(tree):
    seq = ['{', str(tree[0])]
    for child in tree[1]:
        seq.append(cvt_tree_to_bracket_seq(child))
    seq.append('}')
    return ''.join(seq)

def cvt_tree_to_int_seq(tree, left_bracket=1, right_bracket=2):
    """
    Convert a tree to a sequence of intergers, like the function cvt_tree_to_bracket_seq(tree).
    """
    seq = [left_bracket, tree[0]]
    for child in tree[1]:
        seq.extend(cvt_tree_to_int_seq(child, left_bracket, right_bracket))
    seq.append(right_bracket)
    return seq

def tree_distance(tree_a, tree_b):
    seq_a = cvt_tree_to_bracket_seq(tree_a)
    seq_a = helpers.Tree.from_text(seq_a)
    seq_b = cvt_tree_to_bracket_seq(tree_b)
    seq_b = helpers.Tree.from_text(seq_b)
    apted = APTED(seq_a, seq_b)
    distance = apted.compute_edit_distance()
    return distance

def layout_similarity(layout_a, layout_b):
    distance = tree_distance(layout_a, layout_b)
    n = max(count_tree_nodes(layout_a), count_tree_nodes(layout_b))
    return 1 - distance / n
