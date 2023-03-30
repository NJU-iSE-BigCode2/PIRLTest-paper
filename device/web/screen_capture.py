import os, sys
sys.path.append(os.getcwd())

from web.tool_base import *
from logger import logger


class WebScreenCapturer(WebToolBase):
    def __init__(self, driver):
        super(WebScreenCapturer, self).__init__(driver=driver)

    def capture(self, path):
        try:
            self.driver.minimize_window()
            self.driver.maximize_window()
            picture = self.driver.get_screenshot_as_file(path)
            if picture:
                return True
            else:
                logger.error("fail to capture the screen")
                return False
        except Exception as e:
            logger.error(e)
            return False

def main():
    driver = webdriver.Chrome()
    web_cap = WebScreenCapturer(driver)
    web_cap.driver.get("http://www.baidu.com")
    if not os.path.exists("./pics/"):
        os.mkdir("./pics/")
    web_cap.capture('./pics/tmp.png')
    driver.quit()


if __name__ == "__main__":
    main()
