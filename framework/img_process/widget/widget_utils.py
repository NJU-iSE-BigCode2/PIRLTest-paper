from img_process.widget.widgets_embedding import embed_img
import numpy as np


def iou_score(a_bbox, b_bbox):
    x1, y1, w1, h1 = a_bbox
    x2, y2, w2, h2 = b_bbox
    total = w1 * h1 + w2 * h2
    intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union = total - intersection
    return intersection / union

def widget_bbox_match(src_bboxes, tgt_bboxes, threshold=.7):
    if len(src_bboxes) == 0 or len(tgt_bboxes) == 0:
        return []
    matches = []
    for i, src_bbox in enumerate(src_bboxes):
        best_j, _ = max(enumerate(tgt_bboxes), 
                        key=lambda v: iou_score(src_bbox, v[1]))
        best_score = iou_score(src_bbox, tgt_bboxes[best_j])
        if best_score >= threshold:
            matches.append((i, best_j))
    return matches

def widget_similarity(src_image, src_bboxes, 
                      tgt_image, tgt_bboxes,
                      bbox_threshold=.7):
    if len(src_bboxes) == 0 and len(tgt_bboxes) == 0:
        return 1, []
    if len(src_bboxes) == 0 or len(tgt_bboxes) == 0:
        return 0, []

    matches = widget_bbox_match(src_bboxes, tgt_bboxes, bbox_threshold)
    if matches:
        matched_src_bboxes = [src_bboxes[i] for i, _ in matches]
        matched_src_widgets = [src_image[y:y+h, x:x+w, :] for x, y, w, h in matched_src_bboxes]
        matched_src_embeddings = embed_img(matched_src_widgets)
        matched_tgt_bboxes = [tgt_bboxes[j] for _, j in matches]
        matched_tgt_widgets = [tgt_image[y:y+h, x:x+w, :] for x, y, w, h in matched_tgt_bboxes]
        matched_tgt_embeddings = embed_img(matched_tgt_widgets)
        matching_distance = np.sum(np.clip(np.sqrt(np.sum((matched_src_embeddings - matched_tgt_embeddings) ** 2, axis=1)), 0, 1))
    else:
        matching_distance = 0

    remaining_bboxes = [src_bboxes[i] for i in set(j for j in range(len(src_bboxes))) - set(j for j, _ in matches)]
    if remaining_bboxes:
        remaining_widgets = [src_image[y:y+h, x:x+w, :] for x, y, w, h in remaining_bboxes]
        remaining_embeddings = embed_img(remaining_widgets)
        remaining_distance = np.sum(np.clip(np.sqrt(np.sum(remaining_embeddings ** 2, axis=1)), 0, 1))
    else:
        remaining_distance = 0
    sim = 1 - (matching_distance + remaining_distance) / len(src_bboxes)
    return sim, matches

