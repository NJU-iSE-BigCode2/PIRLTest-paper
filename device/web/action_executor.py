from web.tool_base import *
from selenium.webdriver.common.action_chains import ActionChains
import time
from logger import logger
import re


class WebExecutor(WebToolBase):
    def __init__(self, driver, home_url, no_back_on_url=None):
        super(WebExecutor, self).__init__(driver=driver)
        self.driver.maximize_window()
        self.all_h = self.driver.window_handles
        self.home_url = home_url
        self.no_back_on_url = no_back_on_url

    def _switch_handle(self, handle=None):
        if handle:
            self.driver.switch_to.window(handle)
        else:
            if len(self.driver.window_handles) > len(self.all_h):
                self.all_h = self.driver.window_handles
                # Stick to the first tab.
                self.driver.switch_to.window(self.all_h[0])

    def _move(self, x, y):
        ActionChains(self.driver).move_by_offset(x, y).perform()

    def left_click(self, *args):
        x = args[0][0] + args[0][2] // 2
        y = args[0][1] + args[0][3] // 2
        ActionChains(self.driver).move_by_offset(x, y).click().perform()
        self._switch_handle() #switch handle to improve speed
        self._move(-x, -y)
        if not self.home_url in self.driver.current_url:
            # Back to the app if jumped to another website.
            self.driver.back()

    def right_click(self, *args):
        x = args[0][0] + args[0][2] // 2
        y = args[0][1] + args[0][3] // 2
        ActionChains(self.driver).move_by_offset(x, y).context_click().perform()
        self._switch_handle()
        self._move(-x, -y)

    def write(self, *args):
        """first click second input"""
        x = args[0][0] + args[0][2] // 2
        y = args[0][1] + args[0][3] // 2
        ActionChains(self.driver).move_by_offset(x, y).click().send_keys('test').perform()
        self._switch_handle()
        self._move(-x, -y)

    def _can_back_on(self, url):
        if url == self.home_url or url == f'{self.home_url}/':
            return False
        if self.no_back_on_url and re.match(self.no_back_on_url, url):
            return False
        return True

    def back(self):
        """return"""
        url = self.driver.current_url
        if self._can_back_on(url):
            self.driver.back()
        else:
            logger.debug(f'No back on {url}.')
        time.sleep(.5)
        if not self.home_url in self.driver.current_url:
            # Back to home.
            self.driver.get(self.home_url)

    def scroll(self):
        """half page down"""
        js = f"window.scrollBy(0, document.body.clientHeight / 2)"
        self.driver.execute_script(js)

    def exec_action(self, action):
        methods = [
            self.left_click,
            None,
            None,
            self.write,
            self.scroll,
            None,
            self.right_click,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.back,
        ]
        method = methods[action[0]]
        if method is not None:
            return method(*action[1:])
        raise ValueError(f'Action type not supported: {action[0]}.')

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        pass

def main():
    """
    a small test demo
    """
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable - logging'])
    #web = webdriver.Chrome(options=options)
    driver = webdriver.Chrome(options=options)
    web_exe = WebExecutor(driver)
    web_exe.driver.get("http://www.baidu.com")
    web_exe.driver.implicitly_wait(10)
    h = web_exe.driver.current_window_handle
    x,y = 527, 332
    web_exe.left_click(x, y)
    #web_exe.driver.switch_to.window(h)
    web_exe._switch_handle(h)
    x,y = 527+20, 219+0.5
    #web_exe.left_click(x, y)
    web_exe.write("Software Testing", x, y)
    time.sleep(1)

    x,y = 725+0.5, 15+0.5
    web_exe.left_click(x, y)

    time.sleep(1)
    #600 pixals down
    web_exe.scroll(600)
    time.sleep(2)
    #return
    web_exe.back()
    time.sleep(1)
    web_exe.driver.quit()


if __name__ == "__main__":
    main()
