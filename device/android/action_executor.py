from android.tool_base import AndroidToolBase
from logger import logger
import time


class AndroidExecutor(AndroidToolBase):
    def __init__(self, app, screen_size, banner_height, serial_no=None):
        super(AndroidExecutor, self).__init__(serial_no=serial_no)
        self.current_rotation = 0
        self.foreground = True
        self.network_on = True
        self.app = app
        self.screen_size = screen_size
        self.banner_height = banner_height
        self.no_back_acts = app['main_activity'].split(';')

    def tap(self, *args):
        x, y, w, h = args[0]
        y += self.banner_height
        cmd = f'shell input tap {x + w // 2} {y + h // 2}'
        if not self.send_cmd(cmd, 'Error performing tap command.'):
            return False
        time.sleep(0.5)
        if not self.get_current_pkg() == self.app['pkg']:
            return self.back()
        return True

    def _rotate(self, rotation):
        cmd = f'shell settings put system user_rotation {rotation}'
        if self.send_cmd(cmd, 'Error performing rotation.'):
            self.current_rotation = rotation
            return True
        return False

    def rotate(self):
        rotation = 1 if self.current_rotation == 0 else 0
        return self._rotate(rotation)

    def long_press(self, *args):
        bbox = args[0]
        x = bbox[0] + bbox[2] // 2
        y = bbox[1] + bbox[3] // 2 + banner_height
        cmd = f'shell input swipe {x} {y} {x} {y} 1000'
        return self.send_cmd(cmd, 'Error performing long press.')

    def input(self, *args):
        if self.tap(*args):
            cmd = 'shell input text "test"'
            return self.send_cmd(cmd, 'Error inputing text.')
        else:
            return False

    def scroll(self):
        # Scroll down only.
        top = (self.screen_size[0] // 2, self.screen_size[1] // 6 + banner_height)
        bottom = (top[0], 1 - top[1])
        cmd = f'shell input swipe {bottom[0]} {bottom[1]} {top[0]} {top[1]} 500'
        return self.send_cmd(cmd, 'Error scrolling down.')

    def split_screen(self):
        raise NotImplemented

    def switch_back_front(self):
        if self.foreground:
            if self.send_cmd('shell input keyevent 3', 'Error back to home.'):
                self.foreground = False
                return True
            return False
        else:
            cmd = f'shell am start {self.app["pkg"]}'
            if self.send_cmd(cmd, 'Error back to app.'):
                self.foreground = True
                return True
            return False

    def switch_network(self):
        if self.network_on:
            if self.send_cmd('shell svc wifi disable', 'Error turning off wifi.'):
                self.network_on = False
                return True
            return False
        else:
            if self.send_cmd('shell svc wifi enable', 'Error enabling wifi.') and \
                self.send_cmd('shell input keyevent 22', 'Error sending keycode 22.') and \
                self.send_cmd('shell input keyevent 22', 'Error sending keycode 22.') and \
                self.send_cmd('shell input keyevent 66', 'Error sending keycode 66.'):
                self.network_on = True
                return True
            return False

    def grant_permission(self):
        raise NotImplemented

    def deny_permission(self):
        raise NotImplemented

    def interrupt(self):
        raise NotImplemented

    def back(self):
        if not self.get_current_activity() in self.no_back_acts:
            cmd = 'shell input keyevent 4'
            return self.send_cmd(cmd, 'Error pressing back key.')
        # Do nothing on main page to prevent exiting app.
        logger.debug('No back on main page')
        return True

    def get_current_package_activity(self):
        cmd = 'shell dumpsys window windows'
        output = self.get_result(cmd, 'Failed to dumpsys.')
        for line in output.split('\n'):
            if 'mCurrentFocus' in line:
                window = line.split(' ')[-1][:-2]
                if '/' in window:
                    [pkg, act] = window.split('/')
                    return pkg, act
        
        cmd = 'shell dumpsys activity activities'
        output = self.get_result(cmd, 'Failed to dumpsys.')
        for line in output.split('\n'):
            if 'mResumedActivity' in line:
                [pkg, act] = line.split(' ')[-2].split('/')
                if act.startswith('.'):
                    act = pkg + act
                return pkg, act
        return None, None

    def get_current_activity(self):
        return self.get_current_package_activity()[1]

    def get_current_pkg(self):
        return self.get_current_package_activity()[0]

    def exec_action(self, action):
        methods = [
            self.tap,
            None,
            self.long_press,
            self.input,
            self.scroll,
            None,
            None,
            None,
            self.rotate,
            self.split_screen,
            None,
            self.switch_back_front,
            self.switch_network,
            self.grant_permission,
            self.deny_permission,
            self.interrupt,
            self.back
        ]
        method = methods[action[0]]
        if method is not None:
            res = method(*action[1:])
            logger.debug(f'Current activity is {self.get_current_activity()}.')
            return res
        raise ValueError(f'Action type not supported: {action[0]}.')

    def __enter__(self):
        # Disable auto-rotate.
        cmd = 'shell settings put system accelerometer_rotation 0'
        assert self.send_cmd(cmd, 'Error disabling auto-rotate.')

        self._rotate(0)

        return self

    def __exit__(self, type, value, trace):
        # Enable auto-rotate.
        cmd = 'shell settings put system accelerometer_rotation 1'
        assert self.send_cmd(cmd, 'Error enabling auto-rotate.')

