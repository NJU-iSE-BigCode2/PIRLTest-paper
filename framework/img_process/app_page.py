import cv2
import numpy as np
from img_process import canny_ocr
from img_process.layout import layout_extraction as LE
from img_process.layout.tree_embedding_wrapper import tew
from img_process.widget import widgets_embedding as WE
from img_process.layout.layout_utils import layout_similarity, cvt_tree_to_int_seq
from img_process.widget.widget_utils import widget_similarity, widget_bbox_match
from logger import logger


class AppPage:
    def __init__(self, image, bboxes, widget_types, tree, ban_set=None):
        self.image = image
        self.bboxes = bboxes
        self.widget_types = widget_types
        self.tree = tree
        self.ban_set = ban_set if ban_set is not None else set()

    @staticmethod
    def from_image(image_path):
        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape
        bboxes_with_data = canny_ocr.extract(image_path)
        bboxes = [b for [b, _] in bboxes_with_data]
        widgets = [image[y:y+h, x:x+w, :] for x, y, w, h in bboxes]
        #widget_types = WE.classify_widget_type(widgets)
        widget_types = [1 if 'text_len' in d else 0 for [_, d] in bboxes_with_data]
        groups = LE.layout(bboxes, (img_w, img_h))
        tree = LE.generate_tree_struct(groups)
        # Ban rule: If text length is over 5, consider it as a view-only textview, so ban it.
        ban_set = set(b for [b, d] in bboxes_with_data if 'text_len' in d and d['text_len'] > 5)
        return AppPage(image, bboxes, widget_types, tree, ban_set=ban_set)

    @property
    def active_widgets(self):
        return list(set(self.bboxes) - self.ban_set)

    def ban_widget(self, bbox):
        self.ban_set.add(bbox)

    @property
    def all_widgets(self):
        return self.bboxes

    @property
    def layout_embedding(self):
        seq = cvt_tree_to_int_seq(self.tree)
        embedding = np.squeeze(tew.predict(seq), axis=0)
        assert not np.any(np.isnan(embedding)), 'Layout embedding has NaN values.'
        emax = embedding.max()
        emin = embedding.min()
        assert emax != emin, 'All same value in layout embedding.'
        embedding = (embedding - emin) / (emax - emin)
        return 2 * embedding - 1

    @property
    def widget_embedding(self):
        embedding = WE.embed_widgets(self.image, self.bboxes)
        assert not np.any(np.isnan(embedding)), 'Widget embedding has NaN values.'
        emax = embedding.max()
        emin = embedding.min()
        assert emax != emin, 'All same value in widget embedding.'
        embedding = (embedding - emin) / (emax - emin)
        return 2 * embedding - 1

    @property
    def embedding(self):
        return np.concatenate([self.layout_embedding, self.widget_embedding], axis=0)

    def similarity(self, page, layout_weight=.7, bbox_threshold=.7):
        layout_sim = layout_similarity(self.tree, page.tree)
        logger.debug(f'layout similarity: {layout_sim}.')
        widget_sim, _ = widget_similarity(self.image, self.bboxes, page.image, page.bboxes, bbox_threshold=bbox_threshold)
        logger.debug(f'widget similarity: {widget_sim}.')
        return layout_weight * layout_sim + (1 - layout_weight) * widget_sim

    def match(self, page, bbox_threshold=.7):
        return widget_bbox_match(self.bboxes, page.bboxes, threshold=bbox_threshold)

