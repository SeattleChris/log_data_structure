import logging
from os import environ
from google.cloud.logging.handlers import CloudLoggingHandler  # , setup_logging


def config_dict(config, add_to_dict=None):
    """Returns a dict or dict like object (os.environ) with optional updated values.
    Input:
        config: Can be a dict, or None to use os.environ, otherwise uses config.__dict__.
        add_to_dict: Either a dict, or a list/tuple of config class attributes to create a dict.
    Modifies:
        if add_to_dict is given, it may modify the input config dict or os.environ.
    Output:
        A dict, or os.environ (which has dict like methods), updated with values due to optional add_to_dict input.
    """
    if add_to_dict and not isinstance(add_to_dict, dict):  # must be an iterable of config object attributes.
        add_to_dict = {getattr(config, key, None) for key in add_to_dict}
    if config and not isinstance(config, dict):
        config = getattr(config, '__dict__', None)
    if not config:
        config = environ
    if add_to_dict:
        config.update(add_to_dict)
    return config


def _clean_level(level):
    """Used if logging._checkLevel is not available. """
    name_to_level = logging._nameToLevel
    level = _level_to_allowed_num(level, name_to_level)
    if level not in name_to_level.values():
        raise ValueError("The level integer was not a recognized value. ")
    return level


def _level_to_allowed_num(level, name_to_level=logging._nameToLevel):
    """Returns int. Raises ValueError for invalid str (not in name_to_level) or invalid int, or TypeError if needed. """
    max_level = max(name_to_level.values())
    if isinstance(level, str):
        level = name_to_level.get(level.upper(), None)
        if level is None:
            raise ValueError("The level string was not a recognized value. ")
    elif not isinstance(level, int):
        raise TypeError("The level, or default level, must be an appropriate str or int value. ")
    if not 0 <= level <= max_level:
        raise ValueError(f"The level integer ( {level} ) was not a recognized value. Max: {max_level} ")
    return level


def get_named_handler(name="python", logger=logging.root):
    """Returns the CloudLoggingHandler with the matching name attached to the provided logger. """
    try:
        handle = logging._handlers.get(name)
        return handle
    except Exception as e:
        logging.exception(e)
        while logger:
            handlers = getattr(logger, 'handlers', [])
            for handle in handlers:
                if handle.name == name:
                    return handle
            logger = logger.parent
    return None


def move_handlers(source, target, log_level=None):
    """Move all the google.cloud.logging handlers from source to target logger, applying log_level if provided. """
    if not all(isinstance(logger, logging.getLoggerClass()) for logger in (source, target)):
        raise ValueError('Both source and target must be loggers. ')
    stay, move = [], []
    for handler in source.handlers:
        if isinstance(handler, CloudLoggingHandler):
            if log_level:
                handler.level = log_level
            move.append(handler)
        else:
            stay.append(handler)
    if move:
        target.handlers.extend(move)
        source.handlers.clear()
        source.handlers.extend(stay)
    return
