from action_set.action_type import ActionType
from img_process import canny_ocr
from action_set.action_embedding import embed_actions_no_arg, embed_actions_with_widget, embed_actions_with_window
import numpy as np
import cv2
from img_process.widget import widgets_embedding as we

# --------------------------------- General Actions -------------------------------------

def gen_click(widget_bboxes, widget_types):
    return [[ActionType.click, bbox, wtype] for bbox, wtype in zip(widget_bboxes, widget_types)]

def gen_scroll(*args):
    return [[ActionType.scroll]]
    
def gen_input(widget_bboxes, widget_types):
    return [[ActionType.input, bbox, wtype] for bbox, wtype in zip(widget_bboxes, widget_types) if wtype == 1]
    
def gen_switch_back_front(*args):
    return [[ActionType.switch_back_front]]
    
def gen_switch_network(*args):
    return [[ActionType.switch_network]]

# --------------------------------- Android Actions --------------------------------------

def gen_long_press(widget_bboxes, widget_types):
    return [[ActionType.long_press, bbox, wtype] for bbox, wtype in zip(widget_bboxes, widget_types)]
    
def gen_rotate_screen(*args):
    return [[ActionType.rotate_screen]]
    
def gen_split_screen(*args):
    return [[ActionType.split_screen]]
    
def gen_grant_permission(*args):
    return [[ActionType.grant_permission]]
    
def gen_deny_permission(*args):
    return [[ActionType.deny_permission]]
    
def gen_interrupt(*args):
    return [[ActionType.interrupt]]

# --------------------------------- Windows/Web Actions --------------------------------------

def gen_double_click(widget_bboxes, widget_types):
    return [[ActionType.double_click, bbox, wtype] for bbox, wtype in zip(widget_bboxes, widget_types)]
    
def gen_right_click(widget_bboxes, widget_types):
    return [[ActionType.right_click, bbox, wtype] for bbox, wtype in zip(widget_bboxes, widget_types)]
    
def gen_middle_click(widget_bboxes, widget_types):
    return [[ActionType.middle_click, bbox, wtype] for bbox, wtype in zip(widget_bboxes, widget_types)]
    
def gen_drag(widget_bboxes, widget_types):
    return []
    
def gen_resize_window(*args):
    sizes = [(i + 1) / 10 for i in range(10)]
    return [[ActionType.resize_window, size] for size in sizes]

def gen_back(*args):
    return [[ActionType.back]]
    
generators = [
    gen_click,
    gen_double_click,
    gen_long_press,
    gen_input,
    gen_scroll,
    gen_drag,
    gen_right_click,
    gen_middle_click,
    gen_rotate_screen,
    gen_split_screen,
    gen_resize_window,
    gen_switch_back_front,
    gen_switch_network,
    gen_grant_permission,
    gen_deny_permission,
    gen_interrupt,
    gen_back,
]

# ----------------------------------- Generate Action Set ---------------------------------
def get_generators(action_types):
    return [generators[i] for i in action_types]

def gen_actions_embeddings(page, sys_info, action_types):
    bboxes = page.active_widgets
    image = page.image
    wtypes = we.classify_widget_type([image[y:y+h, x:x+w, :] for x, y, w, h in bboxes])
    actions = []
    for method in get_generators(action_types):
        actions.extend(method(bboxes, wtypes))
    
    no_arg_actions = [a for a in actions if a[0] in ActionType.no_arg_actions()]
    no_arg_action_embeddings = embed_actions_no_arg(no_arg_actions) if no_arg_actions else None
    widget_actions = [a for a in actions if a[0] in ActionType.widget_actions()]
    widget_action_embeddings = embed_actions_with_widget(widget_actions, image) if widget_actions else None
    window_actions = [a for a in actions if a[0] in ActionType.window_actions()]
    max_size = sys_info['screen_size']
    window_action_embeddings = embed_actions_with_window(window_actions, max_size[0], max_size[1]) if window_actions else None
    actions = no_arg_actions + widget_actions + window_actions
    embeddings = np.concatenate(list(filter(lambda k: k is not None, [
        no_arg_action_embeddings, 
        widget_action_embeddings,
        window_action_embeddings,
    ])), axis=0)
    emax = np.tile(np.expand_dims(embeddings.max(axis=1), 1), (1, embeddings.shape[1]))
    emin = np.tile(np.expand_dims(embeddings.min(axis=1), 1), (1, embeddings.shape[1]))
    embeddings = 2 * (embeddings - emin) / (emax - emin) - 1

    return actions, embeddings
