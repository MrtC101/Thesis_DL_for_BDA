import time
from utils.loggers.console_logger import LoggerSingleton

log = LoggerSingleton()


def measure_time(func):
    """
    Decorador que mide el tiempo de ejecución de una función y registra los resultados.

    Args:
        func (callable): La función a ser cronometrada.

    Returns:
        callable: La función decorada que mide el tiempo de ejecución.
    """

    def wrapper(*args, **kwargs):
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

    return wrapper
