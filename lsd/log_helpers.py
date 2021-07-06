import logging
from .cloud_log import NON_EXISTING_LOGGER_NAME, LowPassFilter, IgnoreFilter  # , CloudParamHandler
from google.cloud.logging.handlers import CloudLoggingHandler  # , setup_logging


def reduce_range_overlaps(ranges):
    """Given a list with each element is a 2-tuple of min & max, returns a similar list simplified if possible. """
    ranges = [ea for ea in ranges if ea]
    if len(ranges) < 2:
        return ranges
    first, *ranges_ordered = list(reversed(sorted(ranges, key=lambda ea: ea[1] - ea[0])))
    r_min = first[0]
    r_max = first[1]
    disjointed_ranges = []
    for r in ranges_ordered:
        if r_min <= r[0] <= r_max:
            r_max = max(r[1], r_max)
        elif r_min <= r[1] <= r_max:
            r_min = min(r[0], r_min)
        # Since we already looked at 'first' sorted by max range, not possible: r[0] < r_min and r[1] > r_max
        # elif r[0] == r[1]:
        #     pass
        else:  # range is possibly disjointed from other ranges. There may be a gap.
            disjointed_ranges.append(r)
    big_range = (r_min, r_max)
    clean_ranges = [big_range, *disjointed_ranges]
    return clean_ranges  # most commonly a list of 1 tuple. Multiple tuples occur for disjointed ranges.


def determine_filter_ranges(filters, name, low_level):
    """For a given filters, determine the ranges covered for LogRecord with given name. """
    max_level = logging.CRITICAL + 1  # Same as logging.FATAL + 1
    if not isinstance(filters, (list, tuple, set)):
        filters = [filters]
    if not len(filters):
        r = (low_level, max_level)
        return [r]
    low_name_match = ['', name]
    temp = name
    for _ in range(0, name.count('.')):
        temp = temp.rpartition('.')[0]
        low_name_match.append(temp)
    low_name_match.append(NON_EXISTING_LOGGER_NAME)
    ranges = []
    for filter in filters:
        if isinstance(filter, LowPassFilter) and name in filter._allowed:
            r = (low_level, max_level)
        elif isinstance(filter, LowPassFilter) and filter.name in low_name_match:
            r = (low_level, filter.below_level)
        elif isinstance(filter, IgnoreFilter) and name in filter.ignore:
            r = tuple()
        elif isinstance(filter, logging.Filter) and getattr(filter, 'name', '') not in low_name_match[:-1]:
            r = tuple()
        else:
            r = (low_level, max_level)
        ranges.append(r)
    return ranges


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
        source.handlers = stay
    return


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
