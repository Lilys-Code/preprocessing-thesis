import time


def time_function(func, *args, **kwargs):
    """Call `func` with the given arguments and return its result alongside elapsed time.

    Returns a (result, elapsed_seconds) tuple. Used throughout the experiment
    runner to measure preprocessing and training wall-clock time without
    modifying the functions being timed.
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed