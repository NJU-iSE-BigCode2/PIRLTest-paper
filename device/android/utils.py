import subprocess
from logger import logger


def send_adb_cmd(cmd, err_msg, stderr=True):
    proc = subprocess.run(cmd.split(), capture_output=True)
    if proc.returncode:
        logger.error(f'{err_msg}: See the details below.')
        logger.error((proc.stderr if stderr else proc.stdout).decode().strip())
        return False
    return True

def get_adb_result(cmd, err_msg):
    proc = subprocess.run(cmd.split(), capture_output=True)
    if proc.returncode:
        logger.error(f'{err_msg}: See the details below.')
        logger.error(proc.stderr.decode().strip())
        return None
    return proc.stdout.decode().strip()

def list_devices():
    cmd = f'adb devices'
    result = get_adb_result(cmd, 'Error list devices.')
    if result:
        devices = []
        lines = result.split('\n')[1:]
        for line in lines:
            devices.append(line.split()[0])
        return devices
    return None

