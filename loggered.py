import os
import subprocess
import random
import logging
import time
import os
from datetime import timedelta

_logger = None


'''

create a path to store log

'''
def get_dump_path(main_dump_path, exp_name):
    assert len(exp_name) > 0
    # create the sweep path if it does not exist
    if not os.path.isdir(main_dump_path):
        subprocess.Popen("mkdir %s" % main_dump_path, shell=True).wait()
    sweep_path = os.path.join(main_dump_path, exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir %s" % sweep_path, shell=True).wait()
    # randomly generate a experiment ID
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        folder_name = ''.join(random.choice(chars) for _ in range(10))
        dump_path = os.path.join(sweep_path, folder_name)
        if not os.path.isdir(dump_path):
            break
    # create the dump folder
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir %s" % dump_path, shell=True).wait()
    return dump_path


class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s [%d] - %s - %s" % (
            record.levelname,
            os.getpid(),
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def get_logger(filepath=None):
    """
    Create a logger.
    """
    global _logger
    if _logger is not None:
        assert _logger is not None
        return _logger
    assert filepath is not None
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    log_formatter = LogFormatter()
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    _logger = logger
    return logger
