from action_set.action_type import ActionType
import numpy as np
from img_process.widget import widgets_embedding as we


def embed_action_type(action_types):
    return np.eye(ActionType.count())[action_types]

def embed_window_size(window_sizes, max_width, max_height):
    sizes_vector = np.expand_dims(np.array(window_sizes), axis=1)
    base_vector = np.array([[max_width, max_height, 1]])
    return sizes_vector * base_vector

def embed_actions_no_arg(actions, type_embed_size=200):
    action_types = [a[0] for a in actions]
    atype_embeddings = embed_action_type(action_types)
    # type_embed_size is for widget type.
    remainings = np.zeros((len(actions), 4096 + 4096 + type_embed_size + 3))
    return np.concatenate([atype_embeddings, remainings], axis=1)

def embed_actions_with_widget(actions, image, type_embed_size=200):
    atype_embeddings = embed_action_type([a[0] for a in actions])
    bboxes = [a[1] for a in actions]
    widget_img_embeddings = we.embed_img([image[y:y+h, x:x+w, :] for x, y, w, h in bboxes])
    widget_bbox_embeddings = we.embed_bbox(bboxes, image.shape[1], image.shape[0])
    widget_type_embeddings = np.eye(type_embed_size)[[a[2] for a in actions]]
    remainings = np.zeros((len(actions), 3))
    return np.concatenate([atype_embeddings, widget_img_embeddings, 
                           widget_bbox_embeddings, widget_type_embeddings, 
                           remainings], axis=1)

def embed_actions_with_window(actions, max_width, max_height, type_embed_size=200):
    atype_embeddings = embed_action_type([a[0] for a in actions])
    widget_embeddings = np.zeros((len(actions), 4096 + 4096 + type_embed_size))
    window_embeddings = embed_window_size([a[1] for a in actions], max_width, max_height)
    return np.concatenate([atype_embeddings, widget_embeddings, window_embeddings], axis=1)
