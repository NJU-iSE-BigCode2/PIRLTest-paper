import logging


logging.basicConfig()
logger = logging.getLogger('pirltest')

def init_logger(filepath, mode='a'):
    logger.setLevel(logging.DEBUG)

    fmt = '%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    fout = logging.FileHandler(filepath, mode=mode)
    fout.setLevel(logging.DEBUG)
    fout.setFormatter(formatter)
    logger.addHandler(fout)

    # Do not propagete message to its ancestors.
    logger.propagate = False

    logger.info('Logger inited.')

init_logger('pirltest-backend.log', 'w')
