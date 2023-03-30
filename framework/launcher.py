import numpy as np
import tensorflow as tf
import cv2
from img_process.app_page import AppPage
from img_process.pop_up_detection import detect_pop_up
from action_set.action_type import ActionType
from action_set.actions import gen_actions_embeddings
from strategy.qnet_wrapper import QNetWrapper
from strategy.strategy import BoltzmannStrategy
from replay_memory import ReplayMemory
from rewards import PageMemory, PageTransitionMemory, generate_reward
from logger import logger


class BackendConfig:
    def __init__(self):
        # -------- Q Net Configs ------------
        self.state_action_size = 17316
        # available args:
        # - hidden_size: size of hidden layers
        # - num_layers: number of hidden layers
        # - activation_fn: activation function of hidden layers
        # - max_value: max output value, should be at least 1 / (1 - gamma)
        self.qnet_args = dict(max_value=2)
        self.initial_lr = .001
        self.max_grad = 1
        self.ckpt_dir = 'ckpts/qnet'
        self.qnet_model_name = 'q-net'
        self.qnet_gpu_fraction = 1.0

        # -------- Training & Predicting Args -----------
        self.train_every_pred = 1  # train the model every ? predictions
        self.train_batch_size = 16
        self.num_train_data = 16
        self.num_train_epoch = 2
        self.pred_batch_size = 16
        self.epsilon = .2
        self.num_warm_ups = 16 * 2

        # -------- Action Configs ------------------
        self.action_types = [[
            ActionType.tap,
            ActionType.back,
        ], [
            ActionType.click,
            ActionType.back,
        ], [
            ActionType.click,
        ]]
        self.action_type_coefficients = [
            1, 1, 1, 1,
            1, 1, 1, 1,
            .5, 1, 1, 1,
            1, 1, 1, 1,
            1,
        ]

        # -------- Replay Memory Configs -------------
        self.gamma = .5  # see also self.qnet_args
        self.rm_size_limit = 1 << 30  # 1GB
        self.rm_dump_dir = 'replay-memory'


class BackendLauncher:
    def __init__(self, config=None):
        self.config = BackendConfig() if config is None else config
        self.last_page = None
        self.last_sa = None
        self.last_action = None
        self.strategy = BoltzmannStrategy(
            qnet_wrapper_args=dict(
                state_action_size=self.config.state_action_size, 
                net_args=self.config.qnet_args,
                lr=self.config.initial_lr,
                max_grad=self.config.max_grad,
                ckpt_dir=self.config.ckpt_dir,
                model_name=self.config.qnet_model_name,
                session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=self.config.qnet_gpu_fraction
                ))
            ),
            batch_size=self.config.pred_batch_size,
            action_type_coefficients=self.config.action_type_coefficients,
            # Args below are not used by BoltzmannStrategy.
            # Uncomment them when using EpsilonGreedyStrategy.
            #epsilon=self.config.epsilon,
            #num_warm_ups=self.config.num_warm_ups,
        )
        self.prediction_counter = 0
        self.page_memory = PageMemory()
        self.ptm = PageTransitionMemory()
        self.replay_memory = ReplayMemory(self.ptm,
                                          size_limit=self.config.rm_size_limit,
                                          dump_dir=self.config.rm_dump_dir,
                                          gamma=self.config.gamma)
        self.sys_info = dict()

    def update_sys_info(self, sys_info):
        self.sys_info.update(**sys_info)

    def _get_current_page(self, screenshot_path, use_pop_up=True):
        page = AppPage.from_image(screenshot_path)
        if is_page_still(page, self.last_page):
            # The widget of last action is inactive. So ban it.
            if self.last_action[0] in ActionType.widget_actions():
                page.ban_widget(self.last_action[1])

        if use_pop_up:
            pop_up_bbox = detect_pop_up(page.image)
            if pop_up_bbox is not None:
                logger.info('Pop-up detected.')
                x1, y1, x2, y2 = pop_up_bbox
                dot_i = screenshot_path.rfind('.')
                pop_up_path = f'{screenshot_path[:dot_i]}--popup{screenshot_path[dot_i:]}'
                cv2.imwrite(pop_up_path, page.image[y1:y2, x1:x2, :])
                pop_up_page = AppPage.from_image(pop_up_path)
                # Apply offset to change the coordinates from pop-up space to image space.
                pop_up_page.bboxes = [(x + x1, y + y1, w, h) for x, y, w, h in pop_up_page.bboxes]
                # Use the original image.
                pop_up_page.image = page.image
                if is_page_still(pop_up_page, self.last_page) and self.last_action[0] in ActionType.widget_actions():
                    pop_up_page.ban_widget(self.last_action[1])
                pop_up_page, pop_up_page_confidence = self.page_memory.update(pop_up_page)
                if not len(pop_up_page.active_widgets) == 0:
                    return pop_up_page, pop_up_page_confidence

        page, page_confidence = self.page_memory.update(page)
        return page, page_confidence
    
    def after_observation(self, screenshot_path):
        page, page_confidence = self._get_current_page(screenshot_path)
        state_embedding = page.embedding
        action_types = self.config.action_types[self.sys_info['os_type']]
        actions, action_embeddings = gen_actions_embeddings(page, self.sys_info, action_types)
        logger.info(f'Number of actions generated: {len(actions)}.')
        page.update_actions(actions)

        # Generate reward and save this transition into replay memory.
        if self.last_page is not None:
            last_page = self.last_page
            last_action = self.last_action
            self.ptm.update(last_page, last_action, page)
            # compute reward later
            #reward = generate_reward(last_page, last_action, page, self.ptm)
            page_transition = (last_page, last_action, page, page_confidence)
            self.replay_memory.append(self.last_sa, page_transition, state_embedding, action_embeddings)

        # Find next action and record.
        best_action, best_sa = self.find_best_action(state_embedding, actions, action_embeddings)
        logger.info(f'Next action: {best_action}.')
        page.execute_action(best_action)
        self.last_page = page
        self.last_sa = best_sa
        self.last_action = best_action
        return best_action

    def find_best_action(self, state_embedding, actions, action_embeddings):
        # Train the q-net first.
        if self.prediction_counter > 0 and self.prediction_counter % self.config.train_every_pred == 0:
            train_data = self.replay_memory.get_train_data(self.strategy.qnet_wrapper,
                                                           n=self.config.num_train_data,
                                                           batch_size=self.config.train_batch_size)
            self.strategy.train_qnet(train_data, num_epoch=self.config.num_train_epoch)

        # Do prediction.
        best_action, best_sa_embedding = self.strategy.explore(state_embedding, actions, action_embeddings)
        self.prediction_counter += 1
        return best_action, best_sa_embedding

def is_page_still(page_a, page_b, threshold=.95):
    if page_a is None or page_b is None:
        return False
    image_a = page_a.image
    image_b = page_b.image
    result = cv2.matchTemplate(image_a, image_b, cv2.TM_CCOEFF_NORMED)
    return result[0, 0].tolist() >= threshold


def main():
    import sys, cv2

    if len(sys.argv) < 2:
        logger.error('No input image. Skipping.')
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    assert image is not None, f'Failed to read image: {image_path}.'
    launcher = BackendLauncher()
    launcher.update_sys_info(dict(
        screen_size=(image.shape[1], image.shape[0]),
        os_type=0,  # Android
    ))
    logger.info(launcher.after_observation(image_path))


if __name__ == '__main__':
    main()

