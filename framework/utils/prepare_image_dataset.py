import logging
import requests
import json
import os


def crawl_image(url, save_path, max_trial_count=5):
    for _ in range(max_trial_count):
        r = requests.get(url)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
        else:
            logging.debug(f'Request failure: {url}. Status is {r.status_code}.')
    return False

def prepare_images(json_path, num_images=10, save_dir='drive/MyDrive/urt_data/images'):
    with open(json_path) as f:
        dataset = json.load(f)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    count = 0
    visited_url_set = set()
    for item in dataset:
        url = item['screen_url']
        if url in visited_url_set:
            continue

        visited_url_set.add(url)
        path = os.path.join(save_dir, f'{count}.png')
        if crawl_image(url, path):
            count += 1
            if count == num_images:
                return True
        else:
            logging.warning(f'Failed to download {url}, skipped')
    return False
