from android.tool_base import AndroidToolBase
import cv2


class AndroidScreenCapturer(AndroidToolBase):
    def __init__(self,
                 banner_height,
                 serial_no=None,
                 inner_path='/sdcard/tmp_screenshot.png'):
        super(AndroidScreenCapturer, self).__init__(serial_no=serial_no)
        self.banner_height = banner_height
        self.inner_path = inner_path

    def capture(self, path):
        cmd = f'shell screencap -p {self.inner_path}'
        if not self.send_cmd(cmd, 'Error capturing screen.'):
            return False
        cmd = f'pull {self.inner_path} {path}'
        if not self.send_cmd(cmd, 'Error pulling screenshot file.', stderr=False):
            return False

        # Remove banner.
        img = cv2.imread(path)
        img = img[self.banner_height:]
        cv2.imwrite(path, img)
        return True

def main():
    AndroidScreenCapturer().capture('screenshot.png')


if __name__ == '__main__':
    main()

