"""Logging configuration for crossld.

All diagnostics go to stderr so that stdout stays a clean result stream. The
package logger hierarchy is rooted at ``crossld``; callers (and tests) can attach
their own handlers or use ``caplog`` because :func:`configure_logging` never calls
``logging.basicConfig``.
"""

import logging
import sys

LOGGER_NAME = "crossld"
DEFAULT_LEVEL = logging.INFO
DEFAULT_FORMAT = logging.Formatter(
        '[%(asctime)s] - [%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def configure_logging(level = None, stream=None) -> logging.Logger:
    """Return the package logger, configured once with a stderr handler.

    Repeated calls are idempotent: the level is updated in place and no duplicate
    handlers are added.
    """

    if level is None:
        level = DEFAULT_LEVEL

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    handler = None
    for existing in logger.handlers:
        if getattr(existing, "_crossld_handler", False):
            handler = existing
            break
    if handler is None:
        handler = logging.StreamHandler(sys.stderr if stream is None else stream)
        handler._crossld_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(DEFAULT_FORMAT)
        logger.addHandler(handler)
    handler.setLevel(level)
    return logger
