import sys
import time
import datetime
import json
from client import ssh_exec_cmd, scp_push_file
from plt import create_platform
from logger import logger


def push_sys_info(sys_info, remote_port):
    sys_info_str = json.dumps(sys_info)
    cmd = 'curl -X POST -H "Content-Type: application/json" '\
          f'-d \'{sys_info_str}\' http://localhost:{remote_port}/pirltest-backend/sys-info'
    output, _ = ssh_exec_cmd(cmd)
    output = output.decode()
    if output:
        logger.debug('Output when pushing system info:', output)

def get_next_action(image_path, remote_port):
    cmd = f'curl -G --data-urlencode "image_path={image_path}" http://localhost:{remote_port}/pirltest-backend/next-action'
    output, _ = ssh_exec_cmd(cmd)
    output = output.decode()
    return output

def get_exec_time(func, *args, **kwargs):
    start_t = time.time()
    result = func(*args, **kwargs)
    end_t = time.time()
    seconds = end_t - start_t
    return result, seconds

def get_server_screenshot_name(app_name):
    now = datetime.datetime.now()
    fmt = '%Y%m%d%H%M%S'
    now_str = now.strftime(fmt)
    return f'{app_name}--{now_str}.png'

def launch(config, screenshot_consumer=None, steps=500):
    logger.info(f'Using config: {config}.')
    total_time = config['total_time']

    # Update system info.
    platform = create_platform(config)
    push_sys_info(platform.get_info(), config['remote_port'])

    # Start testing.
    server_proj_dir = config['server_proj_dir']
    server_ss_dir = config['server_screenshot_dir']
    screenshot_path = config['local_screenshot_path']
    capturer = platform.create_screen_capturer()
    max_trials = config['max_trials'] if 'max_trials' in config else 5

    try:
        with platform.create_action_executor() as executor:
            start_t = time.time()
            end_t = start_t + total_time
            scp_cum_time = 0
            #while time.time() - start_t - scp_cum_time < total_time:
            for _ in range(steps):
                try:
                    while True:
                        if capturer.capture(screenshot_path):
                            if screenshot_consumer is not None and callable(screenshot_consumer):
                                screenshot_consumer(screenshot_path)
                            server_ss_name = get_server_screenshot_name(config['app']['name'])
                            server_ss_path = f'{server_proj_dir}/{server_ss_dir}/{server_ss_name}'
                            scp_result, scp_time = get_exec_time(scp_push_file, screenshot_path, server_ss_path)
                            if scp_result:
                                scp_cum_time += scp_time
                                end_t += scp_time
                                action = get_next_action(f'{server_ss_dir}/{server_ss_name}', config['remote_port'])
                                logger.info(f'Next action: {action}.')
                                action = eval(action)
                                if action:
                                    executor.exec_action(action)
                                    time.sleep(.5)
                                    break
                        else:
                            time.sleep(.5)
                except KeyboardInterrupt:
                    break
    finally:
        logger.info(f'Exit when {time.time() - start_t}s has passed. Scp cost {scp_cum_time}s.')
        platform.on_test_ended()

def main():
    # Read config file.
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    else:
        config_path = 'config.json'
    with open(config_path) as f:
        config = json.load(f)
    launch(config)


if __name__ == '__main__':
    main()

