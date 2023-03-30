import cv2
import numpy as np
from img_process.widget.vgg_wrapper import vgg16
from img_process import canny_ocr
from logger import logger


def embed_img(widgets, vgg_input_size=(224, 224), batch_size=None):
    widgets_resized = [cv2.resize(widget / 255, vgg_input_size) for widget in widgets]
    widgets = np.stack(widgets_resized, axis=0)
    embeddings = vgg16.embed(widgets, batch_size=batch_size)
    return embeddings

def embed_bbox(bboxes, img_width, img_height, vgg_input_size=(224, 224), batch_size=None):
    bboxes_resized = [(x * vgg_input_size[0] // img_width,
                       y * vgg_input_size[1] // img_height,
                       w * vgg_input_size[0] // img_width,
                       h * vgg_input_size[1] // img_height) for x, y, w, h in bboxes]
    hot_maps = [np.zeros((*vgg_input_size, 3)) for _ in range(len(bboxes))]
    for hot_map, bbox in zip(hot_maps, bboxes_resized):
        x, y, w, h = bbox
        hot_map[y:y+h, x:x+w, :] = 1
    hot_maps_resized = [cv2.resize(hot_map, vgg_input_size) for hot_map in hot_maps]
    embeddings = vgg16.embed(np.stack(hot_maps_resized, axis=0), batch_size=batch_size)
    return embeddings

def classify_widget_type(widgets):
    # We use 0 for all widgets. Wait for completion.
    return [0] * len(widgets)

def embed_widgets(image, widget_bboxes, type_embed_size=200, batch_size=None):
    if widget_bboxes:
        n = len(widget_bboxes)
        widgets = [image[y:y+h, x:x+w, :] for x, y, w, h in widget_bboxes]
        img_embeddings = embed_img(widgets, batch_size=batch_size)
        bbox_embeddings = embed_bbox(widget_bboxes, image.shape[1], image.shape[0], batch_size=batch_size)
        widget_types = classify_widget_type(widgets)
        type_embeddings = np.eye(type_embed_size)[widget_types]
        features = np.concatenate([img_embeddings, bbox_embeddings, type_embeddings], axis=1)
        result = np.mean(features, axis=0, dtype=np.float32)
        return result
    else:
        return np.zeros((4096 + 4096 + type_embed_size))

def embed_screenshot(image_path, type_embed_size=200, batch_size=None):
    bboxes = canny_ocr.extract(image_path)
    logger.info(f'{len(bboxes)} widgets extracted.')
    image = cv2.imread(image_path)
    return embed_widgets(image, bboxes, type_embed_size, batch_size=batch_size)
