import time
from utils.loggers.console_logger import LoggerSingleton

log = LoggerSingleton()


def measure_time(func, *args, **kwargs):
    """
    Measures the execution time of a function and logs the results beautifully.

    Args:
        func (callable): The function to be timed.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Returns:
        The result of the function being timed.

    Raises:
        TypeError: If `func` is not a callable.
    """

    if not callable(func):
        raise TypeError("`func` must be a callable object (function).")

    start_time = time.perf_counter()  # Use high-resolution timer

    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    # Calculate time components in a more concise and efficient way
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Use f-string for cleaner formatting with appropriate precision
    log_message = f"Function '{func.__name__}' execution time: " + \
        f"{hours:.2f}h | {minutes:.2f}m | {seconds:.2f}s\n" + \
        f"Total seconds: {execution_time:.6f}"
    log.info(log_message)

    return result
