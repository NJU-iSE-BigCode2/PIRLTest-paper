"""
This script wraps the launcher as an web app hosted on localhost.
This is for the convenience of calling its functions.
"""
from flask import Flask, request
from launcher import BackendLauncher
import time
from logger import logger
import traceback


app = Flask(__name__)
launcher = BackendLauncher()

@app.route('/urt-backend/sys-info', methods=['POST'])
def update_sys_info():
    launcher.update_sys_info(request.get_json())
    return ''

@app.route('/urt-backend/sys-info')
def get_sys_info():
    return launcher.sys_info

@app.route('/urt-backend/next-action')
def gen_next_action():
    image_path = request.args.get('image_path')
    logger.debug(f'image_path = {image_path}')
    start_t = time.time()
    try:
        action = launcher.after_observation(image_path)
        end_t = time.time()
        logger.info(f'Computation used {end_t - start_t}s.')
        return str(action)
    except:
        logger.error(traceback.format_exc())
        return str([])

