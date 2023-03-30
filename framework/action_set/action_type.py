class ActionType:
    tap = click = left_click = 0
    double_click = 1
    long_press = 2
    input = 3
    scroll = 4
    drag = 5
    right_click = 6
    middle_click = 7
    rotate_screen = 8
    split_screen = 9
    resize_window = 10
    switch_back_front = 11
    switch_network = 12
    grant_permission = 13
    deny_permission = 14
    interrupt = 15
    back = 16

    @staticmethod
    def count():
        return 17
    
    @staticmethod
    def no_arg_actions():
        return [
            ActionType.scroll, 
            ActionType.switch_back_front, 
            ActionType.switch_network, 
            ActionType.rotate_screen,
            ActionType.split_screen,
            ActionType.grant_permission,
            ActionType.deny_permission,
            ActionType.interrupt,
            ActionType.back,
        ]
    
    @staticmethod
    def widget_actions():
        return [
            ActionType.click,
            ActionType.double_click,
            ActionType.long_press,
            ActionType.right_click,
            ActionType.middle_click,
            ActionType.input,
        ]
    
    @staticmethod
    def window_actions():
        return [
            ActionType.resize_window,
        ]
