'''
IMPORTANT: This script has secret key configured in method `ocr`. DO NOT spread it publicly.
重要：该脚本包含密钥，请勿公开传播。
'''

import os
import sys
import cv2
import json
import numpy as np
from aip import AipOcr
from logger import logger


def canny_boundings(image, canny_sigma=0.33, dilate_count=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower_threshold = int(max(0, (1 - canny_sigma) * v))
    upper_threshold = int(min(255, (1 + canny_sigma) * v))
    img_binary = cv2.Canny(gray, lower_threshold, upper_threshold, -1)
    img_dilated = cv2.dilate(img_binary, None, iterations=dilate_count)
    contours, _ = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find bounding using contours.
    boundings = [cv2.boundingRect(c) for c in contours]
    return boundings


def ocr(image_bytes, lang='CHN_ENG', show_char=False):
    # Use your own id and keys below.
    APP_ID = ''
    API_KEY = ''
    SECRET_KEY = ''

    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    options = {
        'language-type': lang,
        'recognize_granularity': 'small' if show_char else 'big',
        'probability': 'true'
    }
    return client.general(image_bytes, options)


def deep_ocr(image_path, lang='CHN_ENG', prob=0.90, space_ratio=0.9):
    cache_dir = 'algorithm/ocr'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, '%s.json' %
                              os.path.basename(image_path)[0: os.path.basename(image_path).index(".")])
    if os.path.isfile(cache_file):
        result = json.load(open(cache_file, 'r'))
        load_flag = True
    else:
        result = ocr(open(image_path, 'rb').read(), lang, show_char=True)
        load_flag = False

    image = cv2.imread(image_path)
    assert image is not None, 'Cannot read the image file %s.' % image_path
    if 'words_result' not in result:
        print('OCR failed: %s' % str(result), file=sys.stderr)
        return []

    text_boxes = []
    for words in result['words_result']:
        left = words['location']['left']
        top = words['location']['top']
        width = words['location']['width']
        height = words['location']['height']
        if width < 10 or height < 10:
            continue
        split_idx = []
        # No double-ocr here for better performance.
        for i, (p, q) in enumerate(zip(words['chars'][:-1], words['chars'][1:])):
            distance = q['location']['left'] - p['location']['left'] - p['location']['width']
            threshold = space_ratio * min(p['location']['height'], q['location']['height'])
            if distance > threshold:
                split_idx.append(i + 1)
        for i, j in zip([0] + split_idx, split_idx + [len(words['chars'])]):
            texts = words['words'][i:j]
            q_loc = words['chars'][i]['location']
            box_x = q_loc['left']
            box_y = q_loc['top']
            r_loc = words['chars'][j - 1]['location']
            box_w = r_loc['left'] + r_loc['width'] - q_loc['left']
            box_h = q_loc['height']
            text_boxes.append([(box_x, box_y, box_w, box_h), dict(text_len=j-i)])
    if not load_flag:
        json.dump(result, open(cache_file, 'w'), ensure_ascii=False, indent=4)
    return text_boxes


def intersect(rect_a, rect_b):
    a_x, a_y, a_w, a_h = rect_a
    b_x, b_y, b_w, b_h = rect_b
    dx = max(0, min(a_x + a_w, b_x + b_w) - max(a_x, b_x))
    dy = max(0, min(a_y + a_h, b_y + b_h) - max(a_y, b_y))
    S_i = dx * dy
    S_a = a_w * a_h
    S_b = b_w * b_h
    return S_i / S_a, S_i / S_b, S_i / (S_a + S_b - S_i)


def process_bounding(img, boundings, 
                     width_ratio_threshold=0.01,
                     height_ratio_threshold=0.01):
    # Remove blank widgets.
    bboxes = []
    for [bounding, data] in boundings:
        x, y, w, h = bounding
        node = img[y:y + h, x:x + w, :]
        if not np.count_nonzero(node) == 0 and not np.count_nonzero(255 - node) == 0:
            bboxes.append([bounding, data])

    # Remove small widgets.
    img_h, img_w, _ = img.shape
    bboxes = [[b, d] for [b, d] in bboxes if b[2] / img_w >= width_ratio_threshold and b[3] / img_h >= height_ratio_threshold]
    return bboxes


def extract(image_path, threshold=.50):
    image = cv2.imread(image_path)
    assert image is not None, 'Cannot read the image file %s.' % image_path
    boundings = canny_boundings(image)
    ocr_res = []
    try:
        ocr_res = deep_ocr(image_path)
    except Exception as e:
        logger.error(f'Error on deep_ocr: {e}.')
    mods = []
    for [rect_o, data_o] in ocr_res:
        best_match = (None, 0, 0, 0)
        for rect_c in boundings:
            ratio_o, ratio_c, ratio = intersect(rect_o, rect_c)
            if ratio > best_match[3]:
                best_match = (rect_c, ratio_o, ratio_c, ratio)
        rect_c, _, ratio_c, _ = best_match
        if rect_c is not None:
            if ratio_c > threshold:
                mods.append(['replace', rect_c, rect_o, data_o])
                for rect_cc in boundings:
                    if rect_cc == rect_c:
                        continue
                    _, ratio_cc, _ = intersect(rect_o, rect_cc)
                    if ratio_cc > threshold:
                        mods.append(['delete', rect_cc])
            else:
                mods.append(['add', rect_o, data_o])
        else:
            mods.append(['add', rect_o, data_o])

    widgets = {b: dict() for b in boundings}
    for mod in mods:
        if mod[0] == 'replace':
            if mod[1] in widgets:
                del widgets[mod[1]]
                widgets[mod[2]] = mod[3]
            else:
                widgets[mod[2]] = mod[3]
        elif mod[0] == 'add':
            widgets[mod[1]] = mod[2]
        elif mod[0] == 'delete':
            if mod[1] in boundings:
                del widgets[mod[1]]

    bboxes = [[b, d] for b, d in widgets.items()]
    return process_bounding(image, bboxes)


def main():
    if len(sys.argv) < 2:
        print('Usage: python %s <image_path>' % __file__)
        return

    image_path = sys.argv[1]
    bboxes = extract(image_path)
    image = cv2.imread(image_path)
    for x, y, w, h in bboxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite('canny-ocr-result.png', image)


if __name__ == '__main__':
    main()
