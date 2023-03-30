from plt import Platform
from android.screen_capture import AndroidScreenCapturer
from android.action_executor import AndroidExecutor
from android.utils import send_adb_cmd
from android.tool_base import AndroidToolBase
from logger import logger


class PltAndroid(Platform, AndroidToolBase):
    def __init__(self, config):
        Platform.__init__(self, config)
        AndroidToolBase.__init__(self, config['serial_no'])

    def create_screen_capturer(self):
        return AndroidScreenCapturer(self.config['banner_height'], 
                                     serial_no=self.serial_no)

    def create_action_executor(self):
        return AndroidExecutor(self.config['app'], 
                               self.config['screen_size'],
                               self.config['banner_height'],
                               serial_no=self.serial_no)

    def on_test_ended(self):
        pkg = self.config['app']['pkg']
        # Stop app, copy coverage data files and pull them.
        cmd = f'shell am force-stop {pkg}'
        if self.send_cmd(cmd, f'Failed to force stop {pkg}'):
            cmd = 'shell rm -f /sdcard/coverage/*'
            if self.send_cmd(cmd, 'Failed to clear old coverage data'):
                cmd = f"shell run-as {pkg} sh -c 'mv files/coverage*.ec /sdcard/coverage'"
                if self.send_cmd(cmd, 'Failed to move coverage data to sdcard'):
                    cmd = f'pull /sdcard/coverage coverages/{self.config["app"]["name"]}'
                    if self.send_cmd(cmd, 'Failed to pull coverage data', stderr=False):
                        return
        logger.error('Coverage data not pulled. Please manually pull them afterwards.')

