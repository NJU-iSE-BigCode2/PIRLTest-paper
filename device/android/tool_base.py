from android.utils import *
from logger import logger


class AndroidToolBase:
    def __init__(self, serial_no=None):
        # Start adb server.
        cmd = 'adb start-server'
        assert send_adb_cmd(cmd, 'Error starting adb server.')

        # Find device if necessary.
        devices = list_devices()
        if serial_no is None:
            if len(devices) == 1:
                self.serial_no = devices[0]
                logger.info(f'Using device: {self.serial_no}')
            elif len(devices) == 0:
                raise RuntimeError('No device found.')
            else:
                raise RuntimeError('Multiple devices found. Please specify one.')
        else:
            if serial_no in devices:
                self.serial_no = serial_no
            else:
                raise RuntimeError(f'Device not found: {serial_no}.')

    def send_cmd(self, cmd, err_msg, stderr=True):
        cmd = f'adb -s {self.serial_no} {cmd}'
        return send_adb_cmd(cmd, err_msg, stderr=stderr)

    def get_result(self, cmd, err_msg):
        cmd = f'adb -s {self.serial_no} {cmd}'
        return get_adb_result(cmd, err_msg)

