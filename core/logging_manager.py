import logging
import colorlog
from asyncio import Queue
from collections import deque
import sys
from colorlog.escape_codes import escape_codes
from logging.handlers import RotatingFileHandler

from core.utils.path_utils import get_data_path

sys.stdout.reconfigure(encoding='utf-8')

escape_codes['orange'] = '\033[38;5;208m'  # 208 是接近橙色的 ANSI 256 色编号

logger_color_mapping = {}

MAX_QUEUE_SIZE = 100


class LogCacheManager:
    def __init__(self):
        self.log_cache: deque = deque(maxlen=MAX_QUEUE_SIZE)
        self.queues: list[Queue] = []

    def add_queue(self):
        que = Queue(maxsize=MAX_QUEUE_SIZE)
        self.queues.append(que)
        return que

    def remove_queue(self, que: Queue):
        self.queues.remove(que)

    def get_cache(self) -> list:
        """Get all logs from cache"""
        return list(self.log_cache)

    def emit(self, time, level, name, message, color):
        # Add to cache
        self.log_cache.append({
            "time": time,
            "level": level,
            "name": name,
            "message": message,
            "color": color
        })
        
        # Send to all queues
        for que in self.queues:
            try:
                que.put_nowait({
                    "time": time,
                    "level": level,
                    "name": name,
                    "message": message,
                    "color": color
                })
            except Exception as e:
                pass


class LogQueueHandler(logging.Handler):
    def __init__(self, log_cache_mgr: LogCacheManager):
        super().__init__()
        self.log_cache_mgr = log_cache_mgr

    def emit(self, record) -> None:
        self.log_cache_mgr.emit(
            record.asctime,
            record.levelname,
            record.name,
            record.message,
            logger_color_mapping.get(record.name, "blue")
        )


log_cache_manager = LogCacheManager()


class GetLoggerFilter(logging.Filter):
    """过滤器：只允许通过get_logger创建的logger的日志通过"""

    def __init__(self, created_loggers_set):
        super().__init__()
        self.created_loggers_set = created_loggers_set

    def filter(self, record):
        # 如果logger名称在created_loggers_set中，则允许通过
        return record.name in self.created_loggers_set


_created_by_get_logger = set()


def get_logger(name: str, color: str):
    logger = logging.getLogger(name)

    _created_by_get_logger.add(name)

    if logger.handlers:
        return logger

    logger_color_mapping[name] = color

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 不传播到上级 logger
    console_formatter = colorlog.ColoredFormatter(
        '%(blue)s%(asctime)s%(reset)s %(log_color)s%(levelname)-8s%(reset)s'
        f'%({color})s[%(name)s]%(reset)s %({color})s%(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'bold_green',
            'WARNING': 'bold_yellow',
            'ERROR': 'bold_red',
            'CRITICAL': 'bold_red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)

    fh = RotatingFileHandler(filename=f"{get_data_path()}/log.log", maxBytes=10*1024*1024, backupCount=1, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)

    qh = LogQueueHandler(log_cache_manager)
    qh.setLevel(logging.DEBUG)
    qh.setFormatter(logging.Formatter(datefmt='%Y-%m-%d %H:%M:%S'))
    qh.addFilter(GetLoggerFilter(_created_by_get_logger))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(qh)
    return logger
