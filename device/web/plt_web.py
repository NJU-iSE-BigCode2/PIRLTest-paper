from plt import Platform
from selenium import webdriver
from web.screen_capture import WebScreenCapturer
from web.action_executor import WebExecutor


class PltWeb(Platform):
    def __init__(self, config):
        super(PltWeb, self).__init__(config)
        options = webdriver.ChromeOptions()
        options.set_capability('unhandledPromptBehavior', 'dismiss')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-notifications')
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(config['app']['url'])
        self.driver.maximize_window()
        if config['pre_interact']:
            input('If pre-interaction is done, press ENTER to continue...')

    def create_screen_capturer(self):
        return WebScreenCapturer(self.driver)

    def create_action_executor(self):
        return WebExecutor(self.driver, 
                           self.config['app']['url'],
                           no_back_on_url=self.config['app'].get('no_back_on_url', None))

    def on_test_ended(self):
        js_code = '''(function(){
            if (window.__coverage__ !== undefined) {
                var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(window.__coverage__));
                var exportName = "out.json";
                var downloadAnchorNode = document.createElement('a');
                downloadAnchorNode.setAttribute("href", dataStr);
                downloadAnchorNode.setAttribute("download", exportName);
                document.body.appendChild(downloadAnchorNode);
                downloadAnchorNode.click();
                downloadAnchorNode.remove();
            }
        })()
        '''
        self.driver.execute_script(js_code)
        # Wait for the coverage file downloaded.
        input('Press ENTER to exit.')
        self.driver.quit()
