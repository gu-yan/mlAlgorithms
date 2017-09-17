#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= protected-access, invalid-name
"""Logging utilities."""
import logging
import sys
import warnings

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET

PY3 = sys.version_info[0] == 3


class _Formatter(logging.Formatter):
    # pylint: disable= no-self-use
    """Customized log formatter."""

    def __init__(self):
        datefmt = '%Y-%m-%d %H:%M:%S'
        super(_Formatter, self).__init__(datefmt=datefmt)

    def _get_color(self, level):
        # pylint: disable= missing-docstring
        if logging.WARNING <= level:
            return '\x1b[31m'
        elif logging.INFO <= level:
            return '\x1b[32m'
        return '\x1b[34m'

    def _get_label(self, level):
        # pylint: disable= missing-docstring
        if level == logging.CRITICAL:
            return 'C'
        elif level == logging.ERROR:
            return 'E'
        elif level == logging.WARNING:
            return 'W'
        elif level == logging.INFO:
            return 'I'
        elif level == logging.DEBUG:
            return 'D'
        return 'U'

    def format(self, record):
        # pylint: disable= missing-docstring
        fmt = self._get_color(record.levelno)
        fmt += self._get_label(record.levelno)
        fmt += ' %(asctime)s --- [%(process)d-%(threadName)s-%(thread)d]'
        fmt += ' %(pathname)s:%(funcName)s:[line:%(lineno)d] :'
        fmt += ']\x1b[0m'
        fmt += ' %(message)s'
        if PY3:
            self._style._fmt = fmt  # pylint: disable= no-member
        else:
            self._fmt = fmt
        return super(_Formatter, self).format(record)


class _Filematter(_Formatter):
    """Customized log formatter."""

    def format(self, record):
        # pylint: disable= missing-docstring
        fmt = self._get_label(record.levelno)
        fmt += ' %(asctime)s --- [%(process)d-%(threadName)s-%(thread)d]'
        fmt += ' %(pathname)s:%(funcName)s:[line:%(lineno)d] :'
        fmt += ' %(message)s'
        if PY3:
            self._style._fmt = fmt  # pylint: disable= no-member
        else:
            self._fmt = fmt
        return super(_Formatter, self).format(record)


def getLogger(name=None, filename=None, filemode=None, level=WARNING):
    """Gets a customized logger.

    .. note:: `getLogger` is deprecated. Use `get_logger` instead.

    """
    warnings.warn("getLogger is deprecated, Use get_logger instead.",
                  DeprecationWarning, stacklevel=2)
    return get_logger(name, filename, filemode, level)


def get_logger(name=None, filename=None, filemode=None, level=WARNING, file_and_line=False):
    """Gets a customized logger.

    Parameters
    ----------
    name: str, optional
        Name of the logger.
    filename: str, optional
        The filename to which the logger's output will be sent.
        If None the logger's output is console line, otherwise file with name
        of 'filename'.
    filemode: str, optional
        The file mode to open the file (corresponding to `filename`),
        default is 'a' if `filename` is not ``None``.
    level: int, optional
        The `logging` level for the logger.
        See: https://docs.python.org/2/library/logging.html#logging-levels
    file_and_line: bool, optional
        Set both console line and file as logger's output,
        default is False. The `filename` will decide the logger's output if False.

    Returns
    -------
    Logger
        A customized `Logger` object.

    Example
    -------
    ## get_logger call with default parameters.
    >>> from mxnet.log import get_logger
    >>> logger = get_logger("Test")
    >>> logger.warn("Hello World")
    W0505 00:29:47 3525 <stdin>:<module>:1] Hello World

    ## get_logger call with WARNING level.
    >>> import logging
    >>> logger = get_logger("Test2", level=logging.WARNING)
    >>> logger.warn("Hello World")
    W0505 00:30:50 3525 <stdin>:<module>:1] Hello World
    >>> logger.debug("Hello World") # This doesn't return anything as the level is logging.WARNING.

    ## get_logger call with DEBUG level.
    >>> logger = get_logger("Test3", level=logging.DEBUG)
    >>> logger.debug("Hello World") # Logs the debug output as the level is logging.DEBUG.
    D0505 00:31:30 3525 <stdin>:<module>:1] Hello World
    """

    logger = logging.getLogger(name)
    if name is not None and not getattr(logger, '_init_done', None):
        logger._init_done = True

        filehdlr = None
        linehdlr = None

        if file_and_line:
            filehdlr = init_filehandler(filename, filemode)
            linehdlr = init_linehandler()
        elif filename:
            filehdlr = init_filehandler(filename, filemode)
        else:
            linehdlr = init_linehandler()

        if filehdlr:
            logger.addHandler(filehdlr)
        if linehdlr:
            logger.addHandler(linehdlr)

        logger.setLevel(level)

    return logger


def init_filehandler(filename, filemode=None):
    """Init a logging filehandler with file mode using formatter ``_Formatter``.

    Parameters
    ----------
    filename: str
        The filename to which the logger's output will be sent.
    filemode: str, optional
        The file mode to open the file (corresponding to `filename`),
        default is 'a'.

    Returns
    -------
    logging.FileHandler
        A `logging.FileHandler` object.
    """
    mode = filemode if filemode else 'a'
    filehdlr = logging.FileHandler(filename, mode)
    filehdlr.setFormatter(_Filematter())
    return filehdlr


def init_linehandler():
    """Init a logging StreamHandler using formatter ``_Filematter``.

    Returns
    -------
    logging.StreamHandler
        A `logging.StreamHandler` object.
    """
    linehdlr = logging.StreamHandler()
    linehdlr.setFormatter(_Formatter())
    return linehdlr
