import time
from logger import logger


def chronometre(func):
    sum_seconds = 0
    n = 0

    def _wrapper(*args, **kwargs):
        start_t = time.time()
        result = func(*args, **kwargs)
        end_t = time.time()
        seconds = end_t - start_t
        nonlocal sum_seconds, n
        sum_seconds += seconds
        n += 1
        logger.debug(f'<{func.__name__}> Moment time cost: {seconds}s, total: {sum_seconds}s, count: {n}.')
        return result

    return _wrapper

