import datetime
import time
from loguru import logger

log_filename = f"process_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
file_handler = logger.add(log_filename)

def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Time taken for {func.__name__}: {elapsed_time:.2f} seconds")
        return result, elapsed_time
    return wrapper
