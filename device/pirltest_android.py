import subprocess
import sys
import time
import json
import launcher
from logger import logger
from android.utils import send_adb_cmd


def start_inst(dev, inst):
    cmd = f'adb -s {dev} shell am instrument {inst}'
    return send_adb_cmd(cmd, f'Failed to start instrumentation: {inst}.')

def main():
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        print('No config file. Exited.')
        return

    with open(config_path) as f:
        config = json.load(f)

    pkg = config['app']['pkg']
    inst = f'{pkg}/.test.JacocoInstrumentation'
    dev = config['serial_no']
    if start_inst(dev, inst):
        time.sleep(5)
        launcher.launch(config)
    else:
        logger.critical('Instrumentation failed.')


if __name__ == '__main__':
    main()
