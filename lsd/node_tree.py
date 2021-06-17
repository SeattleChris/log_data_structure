from collections import defaultdict
import logging
from google.cloud import logging as cloud_logging
from .cloud_log import LowPassFilter
from pprint import pprint

LogClass = logging.getLoggerClass()
MAX_LOG_LEVEL = logging.CRITICAL
MIN_LOG_LEVEL = logging.NOTSET


class LogNode:
    """Helps in constructing relationships between loggers. """

    def __init__(self, logger=None, name='', parent=None, propagate=True, min=MIN_LOG_LEVEL, handlers=None, ):
        self._logger = logger
        self._node_name = name
        self._node_parent = parent  # Expect a LogNode or None.
        self._node_propagate = propagate
        self._node_min = min
        self.add_handlers(handlers, on_node=True)  # Also sets self._updated_handlers = True
        self.children, self._handlers = set(), set()
        self.ranges, self.cached_len = [], 0
        self._name, self._parent, self._propagate = None, None, None
        self._logger_min, self._min, self._max = None, None, None

    @property
    def logger_min(self):
        """Sets level on first access of attached logger with non-zero level. Returns backup value if needed. """
        if not self._logger_min and self._logger:
            self._logger_min = self._logger.level
            if self._logger_min == MIN_LOG_LEVEL:
                self._node_min = MIN_LOG_LEVEL
        return self._logger_min or self._node_min

    @property
    def name(self):
        """Set to logger name on first access after having an attached logger. If no name, returns backup value. """
        if not self._name and self._logger:
            self._name = self._logger.name
        return self._name or self._node_name

    def add_range(self, low, high):
        """Input an inclusive low and an exclusive high (or max level). Return range count, or None if unsuccessful. """
        try:
            assert low < high  # TODO: Consider if equal is okay.
            self.ranges.append((low, high))  # Out of range issues handled in compute_max_min method.
        except AssertionError:
            print(f"Invalid input: {low}, {high} ")
            return None
        except Exception as e:
            logging.exception(e)
            return None
        return len(self.ranges)

    def compute_max_min(self):
        """Updates max & min. Called after confirming the range length is different from the cached range length. """
        self.cached_len = len(self.ranges)
        if self.cached_len == 0:
            self._min, self._max = None, None
            return None
        min_vals = (range[0] for range in self.ranges)
        max_vals = (range[1] for range in self.ranges)
        self._min = max((self.logger_min, min(min_vals)))
        self._max = min(MAX_LOG_LEVEL, max(max_vals))
        return self.cached_len

    @property
    def max(self):
        """Determines the max logging level based on the current ranges. """
        if self._max is None or len(self.ranges) != self.cached_len:
            self.compute_max_min()
        return self._max

    @property
    def min(self):
        """Determines the min logging level based on the current ranges. """
        if self._min is None or len(self.ranges) != self.cached_len:
            self.compute_max_min()
        return self._min


RootLogNode = LogNode(None, None, 'RootNode', False)


def handler_ranges(tree, handlers, log_name, name_low=0, high=MAX_LOG_LEVEL):
    """For a collection of handlers, check for LowPassFilters and determine the logging level ranges of name logs. """
    for handler in handlers:
        handler_low = handler.level
        curr_low = max((name_low, handler_low))
        has_external_log = isinstance((getattr(handler, 'client', None)), cloud_logging.Client)
        if has_external_log:
            if handler.name not in tree:
                handle_node = LogNode(None, handler.name, tree[log_name], False, curr_low)
                tree[handler.name] = handle_node
            tree[log_name].add_child(tree[handler.name])
        name_in_filters = False
        for filter in handler.filters:
            if isinstance(filter, LowPassFilter):
                low = handler_low
                if log_name == filter.name:
                    name_in_filters = True
                    low = curr_low
                if filter.name not in tree:
                    tree[filter.name] = LogNode(None, filter.name, None, False, low)
                tree[filter.name].add_range(low, filter.below_level)
        if not name_in_filters:
            tree[log_name].add_range(curr_low, high)
    return tree


def _walk_logger_tree(node: LogClass or LogNode, tree: dict, name: str = '', name_low: int = MIN_LOG_LEVEL, ):
    if not node:
        return tree
    if not isinstance(node, (LogClass, LogNode)):
        raise ValueError(f"Expected input of None, a LogNode, or a logger: {node} ")
    if not name:
        name = node.name
        name_low = node.min or name_low
    if name not in tree:
        if isinstance(node, LogClass):
            node = LogNode(node)
        tree[name] = node
    tree = handler_ranges(tree, node.handlers, name, name_low)
    tree = _walk_logger_tree(node.parent, tree, name, name_low)
    return tree

    # has_external_log = isinstance((getattr(handler, 'client', None)), cloud_logging.Client)
    # if has_external_log:
    #     name_parent = name
    #     name = handler.name
    #     names.add(name)
    # else:
    #     name_parent = name
    #     name = name

    # name = handler.name if has_external_log else log_name


# curr = self
# while curr:
#     for handler in curr.handlers:
#         has_external_log = isinstance((getattr(handler, 'client', None)), cloud_logging.Client)
#         name = handler.name if has_external_log else log_name
#         names.add(name)
#         grouped_high_level = {name: [] for name in names}
#         high_level = [(f.name, f.below_level) for f in curr.filters if isinstance(f, LowPassFilter) and f.name in names]
#         for name, lvl in high_level:
#             grouped_high_level[name].append(lvl)
#         high_level = max(high_level) if high_level else max_level
#         levels.add((handler.level, high_level))
#     curr = curr.parent if curr.propagate else None
