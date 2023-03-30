import math
from img_process.app_page import AppPage
from img_process.widget.widget_utils import widget_bbox_match, iou_score
from img_process.widget.widgets_embedding import classify_widget_type
from action_set.actions import get_generators
from action_set.action_type import ActionType
from logger import logger


def _is_same_action(a, b, bbox_threshold=.7):
    if not a[0] == b[0]:
        return False
    if a[0] in ActionType.no_arg_actions():
        return True
    if a[0] in ActionType.widget_actions():
        if iou_score(a[1], b[1]) < bbox_threshold:
            return False
        return a[2] == b[2]
    if a[0] in ActionType.window_actions():
        return a[1] == b[1]
    raise RuntimeError(f'Action type unrecognized: {a[0]}.')

class PageWrapper(AppPage):
    def __init__(self, page):
        super(PageWrapper, self).__init__(page.image, page.bboxes, page.widget_types, page.tree, page.ban_set)
        self.extra_bboxes = []
        self.all_actions = []
        self.executed_actions = set()  # indices of all executed actions in self.all_actions
        self.all_bboxes = page.bboxes.copy()

    def merge(self, page, bbox_threshold=.7):
        '''
        Update old page info by new page info and merge widgets.
        '''
        matches = widget_bbox_match(page.all_widgets, self.all_widgets, bbox_threshold)
        extra_indices = list(set(j for j in range(len(page.all_widgets))) - set(j for j, _ in matches))
        extra_bboxes = [page.all_widgets[i] for i in extra_indices]
        self.all_bboxes.extend(extra_bboxes)
        self.bboxes = page.bboxes
        self.widget_types = page.widget_types
        self.tree = page.tree
        self.image = page.image
        self.ban_set.update(page.ban_set)

    def update_actions(self, actions, bbox_threshold=.7):
        '''
        Merge actions.
        '''
        new_actions = []
        for a in actions:
            found = False
            for b in self.all_actions:
                if _is_same_action(a, b, bbox_threshold=bbox_threshold):
                    found = True
                    break
            if not found:
                new_actions.append(a)
        self.all_actions.extend(new_actions)

    def execute_action(self, action, bbox_threshold=.7):
        candidates = [i for i, a in enumerate(self.all_actions) 
                      if _is_same_action(a, action, bbox_threshold=bbox_threshold)]
        if candidates:
            if action[0] in ActionType.widget_actions():
                action_idx = max(candidates, key=lambda c: iou_score(action[1], self.all_actions[c][1]))
            elif len(candidates) == 1:
                action_idx = candidates[0]
            else:
                raise RuntimeError(f'Duplicate action detected: {action}.')
            self.executed_actions.add(action_idx)
        else:
            raise RuntimeError(f'Failed to find action: {action}')

    def match(self, page, bbox_threshold=.7):
        return widget_bbox_match(self.all_widgets, page.all_widgets, bbox_threshold)

    @property
    def all_widgets(self):
        return self.all_bboxes

    def calc_explore_rate(self):
        return len(self.executed_actions) / len(self.all_actions)


class PageMemory:
    def __init__(self):
        self.memory = []

    def update(self, page, threshold=.7):
        if self.memory:
            sims = [page.similarity(p) for p in self.memory]
            best_page, max_sim = max(zip(self.memory, sims), key=lambda t: t[1])
            logger.debug(f'Max similarity: {max_sim}.')
            if max_sim >= threshold:
                logger.debug('Best page found, try merging.')
                best_page.merge(page)
                return best_page, max_sim
        page = PageWrapper(page)
        self.memory.append(page)
        return page, -1


class PageTransitionMemory:
    def __init__(self):
        self.memory = []
        self.counts = []

    def update(self, old_page, action, new_page):
        t = [old_page, action, new_page]
        if t in self.memory:
            self.counts[self.memory.index(t)] += 1
        else:
            self.memory.append(t)
            self.counts.append(1)

    def count(self, old_page, action, new_page):
        t = [old_page, action, new_page]
        return self.counts[self.memory.index(t)] if t in self.memory else 0


def generate_reward(old_page, action, new_page, new_page_confidence, ptm, 
                    punishment_coefficient=3):
    # Core reward.
    er = new_page.calc_explore_rate()
    logger.debug(f'Explore rate: {er}.')
    ptc = ptm.count(old_page, action, new_page)
    logger.debug(f'Page transition count: {ptc}.')
    core_reward = (1 - er) / math.sqrt(ptc)
    # Punishment.
    if old_page == new_page:
        punished_reward = (1 - new_page_confidence) * punishment_coefficient * core_reward
    else:
        punished_reward = core_reward
    # Activation.
    if punished_reward > .5:
        reward = 5 * (math.exp(punished_reward - .5) - 1)
    else:
        reward = 5 * (1 - math.exp(.5 - punished_reward))
    logger.debug(f'Rewards for this action {str(action)}: {core_reward}, {punished_reward}, {reward}')
    return reward

