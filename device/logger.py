import logging
import datetime


logging.basicConfig()
logger = logging.getLogger('pirltest')

def init_logger(filepath, mode='a'):
    logger.setLevel(logging.DEBUG)

    fmt = '%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)

    handlers = [logging.StreamHandler(),
                logging.FileHandler(filepath, mode=mode)]
    for h in handlers:
        h.setLevel(logging.DEBUG)
        h.setFormatter(formatter)
        logger.addHandler(h)

    # Do not propagete message to its ancestors.
    logger.propagate = False

    logger.info('Logger inited.')

def get_now_str():
    now = datetime.datetime.now()
    fmt = '%Y%m%d%H%M%S'
    now_str = now.strftime(fmt)
    return now_str


init_logger(f'logs/pirltest-frontend-{get_now_str()}.log', 'w')
