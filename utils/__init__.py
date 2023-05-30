import functools
import io
import json
import os
from importlib import reload

import pandas as pd

from .parallel import TaskPartitioner


@functools.lru_cache(maxsize=128)
def cached_read_pickle(path):
    return pd.read_pickle(path)


def gpuid_to_device(gpuid, mod=True):
    if isinstance(gpuid, str):
        if gpuid.startswith('cpu') or gpuid.startswith('cuda'): return gpuid
        raise ValueError
    if isinstance(gpuid, list) or isinstance(gpuid, tuple):
        return tuple([gpuid_to_device(_gpuid) for _gpuid in gpuid])
    if gpuid == -1: return 'cpu'
    if gpuid is None: return 'cuda'
    if mod:
        import torch
        gpuid = gpuid % torch.cuda.device_count()
    return 'cuda:%d'%gpuid

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def seed_everything(seed: int):
    if seed is None:
        return
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#========================
import logging


def get_logger(name, log_path, level = logging.INFO, propagate = True):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if len(logger.handlers) > 0 and log_path is None: return logger
    if log_path is not None:
        if not log_path.endswith(".log"):
            raise NotImplementedError()
        log_dir = os.path.dirname(log_path)
        if log_dir == '':
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    log_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    if log_path is not None:
        for handler in [] if len(logger.handlers) == 0 else logger.handlers:
            if os.path.normpath(handler.baseFilename) == log_path:
                break
        else:
            logger.handlers = [] #TODO: This does not make sense in the current case. Maybe change the above to assuming there's only one handler
            fileHandler = logging.FileHandler(log_path, mode='a')
            fileHandler.setFormatter(logFormatter)
            logger.addHandler(fileHandler)
        #logger.warning("\n\nStart of a new log")
    logger.propagate = propagate
    return logger
