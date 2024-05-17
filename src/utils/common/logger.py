from io import TextIOWrapper
import os
import logging
import sys
from tqdm import tqdm
from utils.common.files import is_dir


class TqdmToLog:
    """
    If TqdmToLog is used, the progress bar is printed into a logger,
    braking the tqdm console normal behavior, but allows the tqdm()
    progress bar to be dumpt into a logger file.
    """

    def __init__(self, log: logging.Logger):
        self.log = log

    def write(self, message):
        self.log.info(message)

    def flush(self):
        pass


class LoggerSingleton:
    """
    This module defines a singleton class for a logger. The pipeline utilizes
    only this class' logger, which is modified at the start of each step
    of the pipeline.
    It is possible to change the `name` and the `file handler`
    by simply calling the class constructor with the new arguments.
    This behavior is desirable because all log messages are dumped into one
    file until it is intentionally changed.
    """
    _instance: logging.Logger = None
    _last_file_handler = None
    _level = logging.INFO  # Nivel de registro predeterminado
    _formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d-%Y %H:%M',
    )

    def new_tqdm_handler(cls, folder_path):
        file_name = cls._instance.name.replace(" ", "_").strip("")
        file = os.path.join(folder_path, f"{file_name}.txt")
        tqdm_handler = TqdmLoggingHandler(filename=file, mode='w')
        tqdm_handler.setLevel(cls._level)
        tqdm_handler.setFormatter(cls._formatter)
        cls._instance.addHandler(tqdm_handler)

    def new_console_handler(cls):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(cls._level)
        stream_handler.setFormatter(cls._formatter)
        cls._instance.addHandler(stream_handler)

    def new_file_handler(cls, folder_path):
        is_dir(folder_path)
        file_name = cls._instance.name.replace(" ", "_").strip("")
        file = os.path.join(folder_path, f"{file_name}.txt")
        file_handler = logging.FileHandler(filename=file, mode='a')
        file_handler.setLevel(cls._instance.level)
        file_handler.setFormatter(cls._formatter)
        cls._instance.addHandler(file_handler)
        cls._instance._last_file_handler = file_handler

    def __new__(cls, name=None, level=None, folder_path=None):
        if cls._instance is None:
            cls._instance = logging.getLogger(name)
            cls._instance.setLevel(level or cls._level)
            # cls.new_console_handler(cls)
        if name is not None:
            cls._instance.name = name
        if folder_path is not None:
            if cls._last_file_handler is not None:
                cls._instance.removeHandler(cls._last_file_handler)
            # cls.new_file_handler(cls,folder_path=folder_path)
            cls.new_tqdm_handler(cls, folder_path=folder_path)
        return cls._instance


class TqdmLoggingHandler(logging.Handler):
    """
    The logger class uses only one handler, "TqdmLoggingHandler". This handler
    utilizes the method tqdm.write() for writing into both the standard output
    and the output file. By doing this, you can view each progress with logger
    messages without any problems throw the console.
"""

    def __init__(self, filename, mode):
        super().__init__()
        self.filename = filename
        self.f: TextIOWrapper = open(filename, mode)
        self.mode = mode

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            tqdm.write(msg, file=self.f)
            self.f.flush()
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            self.f.close()
            raise
        except Exception:
            self.f.close()
            self.handleError(record)
