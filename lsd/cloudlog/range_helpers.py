import logging
from .cloud_log import LowPassFilter, IgnoreFilter, CloudParamHandler, StreamTransport


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


def levels_covered(cur_logger, name, level, external=None):
    """Returns a list of range 2-tuples for ranges covered for the named logger traversing from cur_logger.
    Input:
        cur_logger must be a logger instance (CloudLog or other inheriting from logging.Logger).
        name is the initial logger or LogRecord name that is being considered (possibly not same as cur_logger).
        level is the lowest level of concern (usually because the original logger has this level).
        If external is True, only considers LogRecords sent to external (non-standard) log stream.
        If external is False, excludes LogRecords sent to external log streams.
        If external is None (default), reports range(s) sent to some log output.
    Output:
        ranges: List of 2-tuple ranges found affecting 'name' LogRecords when traversing from cur_logger.
        reduced: List of the fewest 2-tuple ranges needed to describe what ranges are covered (often list of a 2-tuple).
    """
    normal_ranges, external_ranges = [], []
    while cur_logger:
        for handler in cur_logger.handlers:
            cur_level = max((handler.level, level))
            handler_ranges = determine_filter_ranges(handler.filters, name, cur_level)
            transport = getattr(handler, 'transport', None)
            if isinstance(handler, CloudParamHandler) and not isinstance(transport, StreamTransport):
                external_ranges.extend(handler_ranges)
            else:
                normal_ranges.extend(handler_ranges)
        cur_logger = cur_logger.parent if cur_logger.propagate else None
    if external:
        ranges = external_ranges
    elif external is None:
        ranges = [*normal_ranges, *external_ranges]
    else:  # external == True
        ranges = normal_ranges
    reduced = reduce_range_overlaps(ranges)
    return ranges, reduced
