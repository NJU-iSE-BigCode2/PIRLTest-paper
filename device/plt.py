class Platform:
    def __init__(self, config):
        self.config = config
        self.pltid = get_platform_id(config['platform'])

    def get_info(self):
        return dict(os_type=self.pltid, 
                    screen_size=self.config['screen_size'])

    def create_screen_capturer(self):
        raise NotImplemented

    def create_action_executor(self):
        raise NotImplemented

    def on_test_ended(self):
        pass

    def __str__(self):
        return self.config['platform']

    def __repr__(self):
        return str(self)

def get_supported_platforms():
    return ['android', 'web']

def get_platform_id(plt_str):
    platforms = get_supported_platforms()
    if plt_str not in platforms:
        raise ValueError(f'Platform not supported: {plt}.')
    return platforms.index(plt_str)

def create_platform(config):
    from android.plt_android import PltAndroid
    from web.plt_web import PltWeb

    plt = config['platform']
    pltid = get_platform_id(plt)
    return [PltAndroid, PltWeb, PltPc][pltid](config)
