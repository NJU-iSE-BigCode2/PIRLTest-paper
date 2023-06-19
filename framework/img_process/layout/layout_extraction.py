'''
This script contains functions about layout characterization

- layout(bounding, resolution): layout characterization
- generate_tree_struct(groups): widget tree generation
- generate_layout_seqs(image_path): from screenshot to widget tree
'''

from img_process import canny_ocr
import cv2
import json


def layout(bounding, resolution): 
    """
        Data Structure Generation
    """
    # Group Generation
    groups = group_generation(basic_row_generation(bounding), resolution)

    # Row & Column Generation.

    line_merge_threshold = 1.5
    column_merge_threshold = 1.5

    for group in groups:
        group[0][0][1] = group[1]
        group[0][-1][2] = group[2]
        #lines = sorted(set([row[1] for row in group[0]] + [row[2] for row in group[0]]))
        nodes = [(y, h) for node_row in group[0] for _, y, _, h in node_row[0]]
        lines = sorted(set([y for y, _ in nodes] + [y + h for y, h in nodes] + [group[1], group[2]]))
        merge_close_lines(lines, line_merge_threshold * resolution[1] / 100)
        lines[0] = group[1]
        lines[-1] = group[2]

        rows = []
        for top, bottom in zip(lines[:-1], lines[1:]):
            filtered_basic_rows = [row for row in group[0] if not (bottom <= row[1] or top >= row[2])]
            filtered_nodes = [(x, w) for row in filtered_basic_rows
                              for x, y, w, h in row[0] if not (y + h <= top or y >= bottom)]
            cols = sorted(set([x for x, _ in filtered_nodes] + [x + w for x, w in filtered_nodes]))
            if len(cols) == 0 or not cols[0] == 0:
                cols = [0] + cols
            if not cols[-1] == resolution[0]:
                cols.append(resolution[0])
            if len(cols) > 0:
                merge_close_lines(cols, column_merge_threshold * resolution[0] / 100)
                cols[0] = 0
                cols[-1] = resolution[0]
                cols = [[left, right] for left, right in zip(cols[:-1], cols[1:])]
            rows.append([cols, top, bottom])
        group[0] = rows

    return groups

def basic_row_generation(bounding):
    basic_rows = []
    for bounding in sorted(bounding, key=lambda b: b[1] + b[3] / 2):
        x, y, w, h = bounding
        center_y = y + h / 2
        found = False
        for row in basic_rows:
            ceiling = row[1]
            ground = row[2]
            if ceiling <= center_y <= ground:
                row[0].append(bounding)
                row[1] = min(ceiling, y)
                row[2] = max(ground, y + h)
                found = True
                break
        if not found:
            basic_rows.append([[bounding], y, y + h])
    return basic_rows

def group_generation(basic_rows, resolution):
    # Initial Group generation.
    groups = [[[row], row[1], row[2]] for row in basic_rows]
    surviving = [True] * len(groups)
    group_count = 0
    while not len(groups) == group_count:
        group_count = len(groups)
        for i, group_i in enumerate(groups):
            for j, group_j in enumerate(groups):
                if not i == j and surviving[j] and \
                        group_j[1] <= (group_i[1] + group_i[2]) / 2 <= group_j[2]:
                    group_j[0] += group_i[0]
                    group_j[1] = min(group_j[1], group_i[1])
                    group_j[2] = max(group_j[2], group_i[2])
                    surviving[i] = False
                    break
    groups = [group for i, group in enumerate(groups) if surviving[i]]

    # Group separation.
    for i, group_i in enumerate(groups):
        for group_j in groups[i + 1:]:
            if group_j[1] < group_i[2] < group_j[2]:
                group_i[2] = int((group_i[2] + group_j[1]) / 2)
                group_j[1] = group_i[2]
            elif group_j[1] < group_i[1] < group_j[2]:
                group_i[1] = int((group_i[1] + group_j[2]) / 2)
                group_j[2] = group_i[1]

    # Group simplification.
    if len(groups) > 0:
        groups[0][1] = 0  # The first group should be at the top.
        groups[-1][2] = resolution[1]  # The last group should be at the bottom.
    for prev, cur in zip(groups[:-1], groups[1:]):
        if prev[2] < cur[1]:
            cur[1] = int((prev[2] + cur[1]) / 2)
            prev[2] = cur[1]
    g_threshold = 1.5 * resolution[1] / 100
    surviving = [True] * len(groups)
    for i in range(len(groups)):
        if groups[i][2] - groups[i][1] < g_threshold:
            if i - 1 < 0 and i + 1 < len(groups):
                groups[i + 1][0] += groups[i][0]
                groups[i + 1][1] = groups[i][1]
            elif i + 1 >= len(groups) and i - 1 >= 0:
                groups[i - 1][0] += groups[i][0]
                groups[i - 1][2] = groups[i][2]
            elif i - 1 >= 0 and i + 1 < len(groups):
                height_a = groups[i - 1][2] - groups[i - 1][1]
                height_b = groups[i + 1][2] - groups[i + 1][1]
                if height_a < height_b:
                    groups[i - 1][0] += groups[i][0]
                    groups[i - 1][2] = groups[i][2]
                else:
                    groups[i + 1][0] += groups[i][0]
                    groups[i + 1][1] = groups[i][1]
            surviving[i] = False
    return [group for i, group in enumerate(groups) if surviving[i]]

def merge_close_lines(lines, threshold=5):
    i = 0
    if len(lines) < 2:
        return
    first = lines[0]
    last = lines[-1]
    while i + 1 < len(lines):
        if lines[i + 1] - lines[i] < threshold:
            lines[i] = int((lines[i] + lines[i + 1]) / 2)
            lines.pop(i + 1)
        else:
            i += 1
    # In case it cannot form a row or a column.
    if len(lines) < 2:
        lines.clear()
        lines.extend([first, last])

def group_json(groups):
    gson = [{
        'rows': [{
            'cols': [{
                'left': col[0],
                'right': col[1]
            } for col in row[0]],
            'top': row[1],
            'bottom': row[2]
        } for row in group[0]],
        'top': group[1],
        'bottom': group[2]
    } for group in groups]
    return json.dumps(gson,indent=2)

def generate_tree_struct(grcs, label=0):
    tree = [label, []]
    children = tree[1]
    for grc in grcs:
        if len(grc) == 3:
            child = generate_tree_struct(grc[0], label)
        else:
            child = [label, []]
        children.append(child)
    return tree

def generate_layout_seqs(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    groups = layout(canny_ocr.extract(image_path), (w, h))
    tree = generate_tree_struct(groups)
    return tree
