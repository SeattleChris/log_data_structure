from collections import defaultdict
import logging
from google.cloud import logging as cloud_logging
from .cloud_log import LowPassFilter
from pprint import pprint
from datetime import time, datetime as dt

LogClass = logging.getLoggerClass()
MAX_LOG_LEVEL = logging.CRITICAL
MIN_LOG_LEVEL = logging.NOTSET
all_nodes = []


class LogNode:
    """Helps in constructing relationships between loggers. """

    def __init__(self, logger=None, name='', parent=None, propagate=True, min=MIN_LOG_LEVEL, handlers=None, ):
        self.children, self._handlers, self._node_handlers = set(), set(), set()
        self.ranges, self.cached_len, self._compute_handlers = [], 0, True
        self._name, self._parent, self._propagate = None, None, None
        self._logger_min, self._min, self._max = None, None, None
        self._logger = logger
        self._node_name = name
        self._node_parent = parent  # Expect a LogNode or None.
        self._node_propagate = propagate
        self._node_min = min
        self.add_handlers(handlers, on_node=True)  # Also sets self._compute_handlers = True
        self.created = dt.now().isoformat()
        all_nodes.append(self)

    @property
    def handlers(self):
        """Returns _handlers combined with either retrieved _logger handlers or _node_handlers. """
        if self.compute_handlers:
            handlers = self._cache_handlers if self._logger else self._node_handlers
            self._handlers = handlers.union(self._handlers)
            self.compute_handlers = 'reset'
        return self._handlers

    def add_handlers(self, handlers, on_node=False):
        """Takes an iterator of handlers and adds them to those considered affecting this LogNode. """
        source = '_node_handlers' if on_node else '_handlers'
        initial = getattr(self, source, set())
        if handlers is None:
            handlers = set()
        elif isinstance(handlers, logging.Handler):
            handlers = [handlers]  # Put the individual one in a list.
        if not isinstance(handlers, (list, tuple, set)):
            raise TypeError("Expected handlers to be a collection that can be cast to a Set of individual objects. ")
        handlers = initial.union(handlers)
        if initial != handlers:
            self._compute_handlers = True
            setattr(self, source, handlers)
        return handlers  # or return self.handlers to include computed handlers from _logger?

    @property
    def compute_handlers(self):
        """Decides to shortcut or compute handlers. Stays True until set to 'reset'. Updates handlers until 'clear'. """
        if self._logger:
            current = set(self._logger.handlers)
            previous = getattr(self, '_cache_handlers', set())
            if current != previous:
                self._cache_handlers = current
                self._compute_handlers = True
        return self._compute_handlers

    @compute_handlers.setter
    def compute_handlers(self, update):
        if update == 'reset':
            self._compute_handlers = False
        elif update == 'clear':
            self._handlers = set()
            self._compute_handlers = True
        else:
            self._compute_handlers = self._compute_handlers or bool(update)

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

    @property
    def parent(self):
        """A LogNode for the parent logger, may create it or use RootLogNode. May set current as child of parent. """
        if not self._parent:
            if not self._logger:  # The self._parent stays as None. May return None, _node_parent, or RootLogNode.
                use_root_node = self._node_parent and not self.propagate
                return RootLogNode if use_root_node else self._node_parent
            temp_parent = self._logger.parent or RootLogNode
            temp_propagate = self.setup_propagate(self._logger.propagate)
            if not self._node_parent:
                self._node_propagate = temp_propagate
                self._node_parent = self.setup_parent(temp_parent, temp_propagate)
            if self._node_parent and self._node_parent.name == temp_parent.name:
                temp_parent = self._node_parent  # Since _node_parent is a LogNode, it will not create a duplicate.
            else:  # Correct parent and _node_parent are not the same. Modify _<value> and not _node_<value>.
                self._propagate = temp_propagate
            self._parent = self.setup_parent(temp_parent, temp_propagate)
        return self._parent if self.propagate else RootLogNode

    @property
    def propagate(self):
        """The _propagate property is a bool, or None when not computed. Return it or backup value if needed. """
        if self._propagate is not None:
            return self._propagate
        return_value = self._logger.propagate if self._logger else None
        if return_value is None:
            self._propagate, return_value = return_value, self._node_propagate
        else:
            self._propagate = self._node_propagate = return_value  # TODO: Decide if _node_propagate left unchanged.
        return return_value

    @propagate.setter
    def propagate(self, prop):
        self._propagate = self.setup_propagate(prop)

    def setup_propagate(self, prop: bool, force=False):
        """Returns input unmodified, but raises if not a boolean. A connected logger can be changed with force. """
        if not isinstance(prop, bool):
            raise TypeError(f"Did not receive a boolean parameter: {prop} ")
        if force:
            if self._logger:
                self._logger.propagate = prop
                self.reset_propagate()  # Use this to compute it on next access.
                # self._propagate = prop  # Shortcut computing the value since it should arrive to this result.
            else:
                raise ValueError("Can not force propagate without an attached logger. ")
        return prop

    def reset_propagate(self):
        """By unsetting _propagate, the next access to propagate will attempt to compute it. """
        self._propagate = None

    @parent.setter
    def parent(self, node):
        self._node_parent = self.setup_parent(node, add_child=True)

    def setup_parent(self, node, add_child=False):
        if isinstance(node, LogClass):
            node = LogNode(node)
        elif node is None:
            node = RootLogNode
        if not isinstance(node, LogNode):
            raise ValueError(f"Expected a LogNode, logger instance, or None for setting the parent: {node} ")
        if add_child:
            node.add_child(self)
        return node

    def add_child(self, node):
        if isinstance(node, LogClass):
            node = LogNode(node)
        if not isinstance(node, LogNode):
            raise ValueError(f"Excpected a LogNode or logger instance for add_child method: {node} ")
        self.children.add(node)

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

    def __str__(self):
        child_names = ', '.join(child.name for child in self.children)
        parent_name = self.parent.name if self.parent else ''
        return '{} P:{} C{}: {}'.format(self.name, parent_name, len(self.children), child_names)

    def __repr__(self):
        return '<LogNode {}>'.format(self.__str__())


RootLogNode = LogNode(None, 'RootNode', None, False)


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


def get_tree(node: LogClass or LogNode, tree: dict = {}, name: str = '', name_low: int = MIN_LOG_LEVEL, ):
    if node is None:
        return tree
    if not isinstance(node, (LogClass, LogNode)):
        raise ValueError(f"Expected input of None, a LogNode, or a logger: {node} ")
    if node.name not in tree:
        if isinstance(node, LogClass):
            node = LogNode(node)
        tree[node.name] = node
    if not name:  # node can still be a LogClass or LogNode
        name = node.name
        name_low = getattr(node, 'min', getattr(node, 'level', None)) or name_low
    tree = handler_ranges(tree, node.handlers, name, name_low)
    tree = get_tree(node.parent, tree, name, name_low)
    return tree


def make_tree(loggers: list):
    tree = {}
    for logger in loggers:
        tree = get_tree(logger, tree)
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
