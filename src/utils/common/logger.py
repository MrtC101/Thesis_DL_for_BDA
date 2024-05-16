from io import TextIOWrapper
import os
import logging
import sys

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

#TODO primero se crea un singleton de un logger
#TODO despues se crea un handler para ese logger que utiliza la función write de tqdm para escribir en consola y en archivo
#TODO como se quiere registrar los progresos tqdm se hizo la función TqdmToLog que pasa la barra a el log que luego escribe con tqdm que luego escribe en el archivo de salida
    

from utils.common.files import is_dir

class TdqmToLog:
    def __init__(self, log : logging.Logger):
        self.log = log

    def write(self, message):
        self.log.info(message)

    def flush(self):
        pass

class LoggerSingleton:

    _instance: logging.Logger = None
    _last_file_handler = None
    _level = logging.INFO  # Nivel de registro predeterminado
    _formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d-%Y %H:%M',
    )

    def new_tqdm_handler(cls,folder_path):
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
    
    def new_file_handler(cls,folder_path):
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
            #cls.new_console_handler(cls)
        if name is not None:
            cls._instance.name = name
        if folder_path is not None:
            if cls._last_file_handler is not None:
                cls._instance.removeHandler(cls._last_file_handler)
            #cls.new_file_handler(cls,folder_path=folder_path)
            cls.new_tqdm_handler(cls,folder_path=folder_path)
        return cls._instance

class TqdmLoggingHandler(logging.Handler):

    def __init__(self,filename,mode):
        super().__init__()
        self.filename = filename
        self.f :TextIOWrapper = open(filename,mode)
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