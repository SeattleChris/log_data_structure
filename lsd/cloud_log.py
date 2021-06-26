from collections import defaultdict
import logging
import warnings
from sys import stderr, stdout
from flask import json
from google.cloud import logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler  # , setup_logging
from google.cloud.logging_v2.handlers.transports import BackgroundThreadTransport
from google.oauth2 import service_account
from google.cloud.logging import Resource
from os import environ
from datetime import datetime as dt

DEFAULT_FORMAT = logging._defaultFormatter  # logging.Formatter('%(levelname)s:%(name)s:%(message)s')
MAX_LOG_LEVEL = logging.CRITICAL
NON_EXISTING_LOGGER_NAME = 'name-that-does-not-match-any-logger'


def _clean_level(level):
    """Used if logging._checkLevel is not available. """
    name_to_level = logging._nameToLevel
    if isinstance(level, str):
        level = name_to_level.get(level.upper(), None)
        if level is None:
            raise ValueError("The level string was not a recognized value. ")
    elif isinstance(level, int):
        if level not in name_to_level.values():
            raise ValueError("The level integer was not a recognized value. ")
    else:
        raise TypeError("The level, or default level, must be an appropriate str or int value. ")
    return level


class IgnoreFilter(logging.Filter):
    """Allows to pass all log records except those with names in the ignore collection. """

    def __init__(self, name: str = '', ignore: list = list()) -> None:
        super().__init__(name='')
        if name:
            ignore.append(name)
        self.ignore = set(ignore)

    def add(self, name: str):
        """Add another log record name to be ignored and not logged. """
        if not isinstance(name, str):
            raise TypeError("Expected a str input for a log record name. ")
        self.ignore.add(name)
        return True

    def allow(self, name: str):
        """Remove from the ignore collection, allowing log records with that name to be logged. """
        if not isinstance(name, str):
            raise TypeError("Expected a str input for a log record name. ")
        self.ignore.discard(name)
        return True

    def filter(self, record):
        if record.name in self.ignore:
            return False
        return True

    def __repr__(self) -> str:
        return '<IgnoreFilter {}>'.format(', '.join(self.ignore))


class LowPassFilter(logging.Filter):
    """Only allows LogRecords that are exclusively below the specified log level, according to levelno. """
    DEFAULT_LEVEL = logging.WARNING

    def __init__(self, name: str, level: int, title: str = '') -> None:
        super().__init__(name=name)
        self.title = title
        self._allowed = set()
        level = CloudLog.normalize_level(level, self.DEFAULT_LEVEL)
        self.below_level = level
        assert self.below_level > 0

    def allow(self, name):
        """Any level log records with these names are not affected by the low filter. They will always pass through. """
        rv = None
        if isinstance(name, (list, tuple, set)):
            rv = [self.allow(ea) for ea in name]
            name = None
        elif not isinstance(name, str):
            try:
                name = getattr(name, 'name', None)
                assert isinstance(name, str)
                assert name != ''
            except (AssertionError, Exception):
                name = None
        if name and isinstance(name, str):
            self._allowed.add(name)
            rv = name
        if rv is None:
            raise TypeError("Unable to add log record name to the LowPassFilter allowed collection. ")
        return rv

    def filter(self, record):
        if record.name in self._allowed:
            return True
        name_allowed = super().filter(record)  # Returns True if no self.name or if it matches start of record.name
        if not name_allowed or record.levelno > self.below_level - 1:
            return False
        return True

    def __repr__(self):
        name = self.name or 'All'
        if name == NON_EXISTING_LOGGER_NAME:
            name = 'None'
        allowed = ' '
        if len(self._allowed):
            allowed = ' and any ' + ', '.join(self._allowed)
        return '<{} only {} under {}{}>'.format(self.__class__.__name__, name, self.below_level, allowed)


class StreamClient:
    """Substitute for google.cloud.logging.Client, whose presence triggers standard library logging techniques. """
    BASE_CLIENT_PARAMETERS = ('project', 'credentials', '_http', '_use_grpc', 'client_info', 'client_options', )

    def __init__(self, name='', resource=None, labels=None, handler=None, **kwargs):
        base_params = {name: kwargs.pop(name, None) for name in self.BASE_CLIENT_PARAMETERS}
        for key in ('client_info', 'client_options'):
            base_params['_' + key] = base_params.pop(key)
        for key, val in base_params.items():
            setattr(self, key, val)
        # assert kwargs == {}
        self.handler_name = name.lower()
        self.resource = resource or {}
        self.labels = labels or {}
        self._handler = None
        if handler:
            self.handler = handler

    def update_attachments(self, resource=None, labels=None, handler=None):
        """Helpful since the order matters. These may be added to the StreamClient later to assist in management. """
        if isinstance(resource, Resource):
            self.resource = resource
        if isinstance(labels, dict):
            self.labels = labels
        if isinstance(handler, str):
            self.handler_name = handler.lower()
        elif handler:
            self.handler = handler

    def prepare_handler(self, handler_param):
        """Creates or updates a logging.Handler with the correct name and attaches the labels and resource. """
        if isinstance(handler_param, type):
            handler = handler_param()  # handler_param is a logging.Handler class.
        elif issubclass(handler_param.__class__, logging.Handler):
            handler = handler_param  # handler_param is the handler instance we wanted.
        else:  # assume handler_param is None or a stream for logging.StreamHandler
            try:
                handler = logging.StreamHandler(handler_param)
            except Exception as e:
                logging.exception(e)
                raise ValueError("StreamClient handler must be a stream (like stdout) or a Handler class or instance. ")
            handler.set_name(self.handler_name)
        handler.labels = self.labels
        handler.resource = self.resource
        return handler

    @property
    def handler(self):
        """If possible get from weakref created in logging. Otherwise maintain a strong referenced object. """
        rv = self._handler or logging._handlers.get(self.handler_name, None)
        if not rv:
            rv = self.prepare_handler(stderr)
            self.handler = rv
        return rv

    @handler.setter
    def handler(self, handler_param):
        handler = self.prepare_handler(handler_param)
        name = getattr(handler, 'name', None)
        if not name or name not in logging._handlers:
            self._handler = handler
        else:
            self._handler = None  # Forces a lookup on logging._handlers, or creation if not present.

    @property
    def project(self):
        """If unknown, computes & sets from labels, resource, or environ. Raises LookupError if unable to determine. """
        if not getattr(self, '_project', None):
            labels = self.labels  # To only compute once if not yet valid.
            project = labels.get('project_id') or labels.get('project')  # checks resource if labels not valid yet.
            if not project:
                project = environ.get('GOOGLE_CLOUD_PROJECT') or environ.get('PROJECT_ID')
            if not project:
                raise LookupError("Unable to discover the required Project id. ")
            self._project = project
        return self._project

    @property
    def labels(self):
        """If the expected 'project_id' is not in labels, will attempt to get labels from resource or _project. """
        labels_have_valid_data = bool(self._labels.get('project_id', None))
        if not labels_have_valid_data:
            try:
                labels = self.resource.get('labels', {}).copy()
            except Exception:
                labels = {}
            project = labels.get('project_id') or labels.get('project')
            labels['project_id'] = project or self._project
            self._labels.update(labels)  # If all values were None or '', then labels is not yet valid.
        return self._labels

    @labels.setter
    def labels(self, labels):
        if not isinstance(labels, dict):
            raise TypeError("Expected a dict input for labels. ")
        res_labels = self.resource.get('labels', {})
        self._labels = labels = {**res_labels, **labels}

    def logger(self, name):
        """Similar interface of google.cloud.logging.Client, but returns standard library logging.Handler instance. """
        if isinstance(name, str):
            name = name.lower()
        if name != self.handler_name:
            return None
        return self.handler


class StreamTransport(BackgroundThreadTransport):
    """A substitute for BackgroundThreadTransport in CloudParamHandler. Use StreamHandler & StreamClient technique. """

    def __init__(self, client, name, *, grace_period=0, batch_size=0, max_latency=0):  # like BackgroundThreadTransport
        self.client = client
        self.handler_name = name
        self.grace_period = grace_period
        self.batch_size = batch_size
        self.max_latency = max_latency

    def create_entry(self, record, message, **kwargs):
        """Format entry close to (but not exact) the style of BackgroundThreadTransport Worker queue """
        entry = {
            "message": message,
            "python_logger": record.name,
            "severity": record.levelname,
            "timestamp": dt.utcfromtimestamp(record.created),
            }
        entry.update({key: val for key, val in kwargs.items() if val})
        return entry

    def send(self, record, message, **kwargs):
        """Replacing Transport send with similar to logging.StreamHandler.emit with json dict output. """
        entry = self.create_entry(record, message, **kwargs)
        entry = json.dumps(entry)
        try:
            stream = self.stream
            stream.write(entry + self.terminator)  # std library logging issue 35046: merged two stream.writes into one.
            self.flush()
        except RecursionError:  # See standard library logging issue 36272
            raise
        except Exception:
            self.handleError(record)

    @property
    def stream(self):
        return self.handler.stream

    @property
    def handler(self):
        return self.client.logger(self.handler_name)

    @property
    def destination(self):
        if not getattr(self, '_destination', None):
            self._destination = self.stream.name.lstrip('<').rstrip('>')
        return self._destination

    @property
    def terminator(self):
        if not getattr(self, '_terminator', None):
            self._terminator = self.handler.terminator
        return self._terminator

    def flush(self):
        self.handler.flush()

    def handleError(self, record):
        return self.handler.handleError(record)

    def __repr__(self) -> str:
        level = logging.getLevelName(self.handler.level)
        return '<%s %s | (%s) %s>' % (self.__class__.__name__, self.destination, level, self.handler_name)


class CloudParamHandler(CloudLoggingHandler):
    """Emits log by CloudLoggingHandler technique with a valid Client, or by StreamHandler if client is None. """

    filter_keys = ('_resource', '_trace', '_span_id', '_http_request', '_source_location', '_labels', '_trace_str',
                   '_span_id_str', '_http_request_str', '_source_location_str', '_labels_str', '_msg_str')

    def __init__(self, client, name='param_handler', resource=None, labels=None, stream=None, ignore=None):
        if not isinstance(client, (StreamClient, cloud_logging.Client)):
            raise ValueError("Expected a StreamClient or cloud logging Client. ")
        transport = StreamTransport if isinstance(client, StreamClient) else BackgroundThreadTransport
        super().__init__(client, name=name, transport=transport, resource=resource, labels=labels, stream=stream)
        # on self: name, transport(client, name), client, project_id, resource, labels; adds a CloudLoggingFilter
        self.ignore = ignore  # self._data_keys = self.get_data_keys(ignore)

    def get_data_keys(self, ignore=None, ignore_str_keys=True):
        """DEPRECATED. Returns a list of the desired property names for logging that are set by CloudLoggingHandler. """
        keys = set(key[1:] for key in self.filter_keys if not (ignore_str_keys and key.endswith('_str')))
        ignore = getattr(self, 'ignore', None) if ignore is None else ignore
        if isinstance(ignore, str):
            ignore = {ignore, }
        if isinstance(ignore, (list, tuple, set)):
            ignore = set(key.lstrip('_') for key in ignore)
        else:
            ignore = set()
        if not hasattr(self, ignore):
            self.ignore = ignore
        keys = keys.difference(ignore)
        return keys

    def prepare_record_data(self, record):
        """Update record attributes set by CloudLoggingHandler and move http_request to labels to assist in logging. """
        resource = getattr(record, '_resource', None)
        if self.resource and not resource:
            record._resource = resource = self.resource
        no_http_req = {'request': 'None. Likely testing local. '}
        http_req = getattr(record, '_http_request', None) or no_http_req
        http_labels = {'_'.join(('http', key)): val for key, val in http_req.items()}
        handler_labels = getattr(self, 'labels', {})
        record_labels = getattr(record, '_labels', {})
        labels = {**http_labels, **handler_labels, **record_labels}
        record._labels = labels
        record._http_request = None
        # print(f"------------------- Prepared: {record.name} ------------------------")
        # print(f"http_request: {record._http_request} ")
        # print(f"Labels: {record._labels} ")
        # print("---------------------------------------------------------")
        if isinstance(self.client, StreamClient) and isinstance(resource, Resource):
            record._resource = resource._to_dict()
        return record

    def emit(self, record):
        """After preparing the record data, will call the appropriate StreamTransport or BackgroundThreadTransport. """
        self.prepare_record_data(record)
        super().emit(record)

    @property
    def destination(self):
        """Keeps a hidden str property that is a cache of previously computed value. """
        if not getattr(self, '_destination', None):
            if isinstance(self.transport, StreamTransport):
                rv = self.transport.destination
            else:
                rv = '-/logs/' + self.name
            self._destination = rv
        return self._destination

    def __repr__(self) -> str:
        level = logging.getLevelName(self.level)
        return '<%s %s | (%s) %s>' % (self.__class__.__name__, self.destination, level, self.name)


class CloudLog(logging.Logger):
    """Extended python Logger class that attaches a google cloud log handler. """
    APP_LOGGER_NAME = __package__  # TODO: Update if this logging package is imported into an application.
    APP_HANDLER_NAME = 'app'
    DEFAULT_LOGGER_NAME = None
    DEFAULT_HANDLER_NAME = None
    DEBUG_LOG_LEVEL = logging.DEBUG
    DEFAULT_LEVEL = logging.INFO
    DEFAULT_HIGH_LEVEL = logging.WARNING
    DEFAULT_RESOURCE_TYPE = 'gae_app'  # 'logging_log', 'global', or any key from RESOURCE_REQUIRED_FIELDS
    LOG_SCOPES = (
        'https://www.googleapis.com/auth/logging.read',
        'https://www.googleapis.com/auth/logging.write',
        'https://www.googleapis.com/auth/logging.admin',
        'https://www.googleapis.com/auth/cloud-platform',
        )
    RESOURCE_REQUIRED_FIELDS = {  # https://cloud.google.com/logging/docs/api/v2/resource-list
        'cloud_tasks_queue': ['project_id', 'queue_id', 'target_type', 'location'],
        'cloudsql_database': ['project_id', 'database_id', 'region'],
        'container': ['project_id', 'cluster_name', 'namespace_id', 'instance_id', 'pod_id', 'container_name', 'zone'],
        # 'k8s_container': RESOURCE_REQUIRED_FIELDS['container']
        'dataflow_step': ['project_id', 'job_id', 'step_id', 'job_name', 'region'],
        'dataproc_cluster': ['project_id', 'cluster_id', 'zone'],
        'datastore_database': ['project_id', 'database_id'],
        'datastore_index': ['project_id', 'database_id', 'index_id'],
        'deployment': ['project_id', 'name'],
        'folder': ['folder_id'],
        'gae_app': ['project_id', 'module_id', 'version_id', 'zone'],
        'gce_backend_service': ['project_id', 'backend_service_id', 'location'],
        'gce_instance': ['project_id', 'instance_id', 'zone'],
        'gce_project': ['project_id'],
        'gcs_bucket': ['project_id', 'bucket_name', 'location'],
        'generic_node': ['project_id', 'location', 'namespace', 'node_id'],
        'generic_task': ['project_id', 'location', 'namespace', 'job', 'task_id'],
        'global': ['project_id'],
        'logging_log': ['project_id', 'name'],
        'logging_sink': ['project_id', 'name', 'destination'],
        'project': ['project_id'],
        'pubsub_subscription': ['project_id', 'subscription_id'],
        'pubsub_topic': ['project_id', 'topic_id'],
        'reported_errors': ['project_id'],
        }
    RESERVED_KWARGS = ('stream', 'fmt', 'format', 'handler_name', 'handler_level', 'res_type', 'parent', 'cred_or_path')
    CLIENT_KW = ('project', 'credentials', 'client_info', 'client_options')  # also: '_http', '_use_grpc'

    def __init__(self, name=None, level=None, automate=False, **kwargs):
        name = self.normalize_logger_name(name)
        level = self.normalize_level(level)
        super().__init__(name, level=level)
        if automate:
            self.automated_structure(**kwargs)

    def automated_structure(self, resource=None, client=None, replace=False, **kwargs):
        """Typically only used for the core logers of the main code. This will add resource, labels, client, etc. """
        # After cleaning out special key-words, the remaining kwargs are used for creating Resource and labels.
        name = self.name
        stream = kwargs.pop('stream', None)
        fmt = kwargs.pop('fmt', kwargs.pop('format', DEFAULT_FORMAT))
        default_handle_name = self.APP_HANDLER_NAME if name == self.APP_LOGGER_NAME else self.DEFAULT_HANDLER_NAME
        default_handle_name = default_handle_name or name
        handle_name = kwargs.pop('handler_name', default_handle_name)
        handle_level = kwargs.pop('handler_level', None)
        res_type = kwargs.pop('res_type', self.DEFAULT_RESOURCE_TYPE)
        parent = kwargs.pop('parent', logging.root)
        self.parent = self.normalize_parent(parent, name)
        cred_or_path = kwargs.pop('cred_or_path', None)
        if client and cred_or_path:
            raise ValueError("Unsure how to prioritize the passed 'client' and 'cred_or_path' values. ")
        client = client or cred_or_path
        client_kwargs = {key: kwargs.pop(key) for key in ('client_info', 'client_options') if key in kwargs}
        labels = getattr(logging.root, '_config_labels', {})
        labels.update(kwargs.pop('labels', None) or kwargs)
        if resource is None:
            resource = getattr(logging.root, '_config_resource', None)
            resource = Resource._from_dict(resource)
        if isinstance(resource, Resource):  # resource may be None, a Config obj, or a dict.
            labels = {**resource.labels, **labels}
        else:
            resource = self.make_resource(resource, res_type, **labels)
            labels = getattr(resource, 'labels', **self.get_environment_labels(), **labels)
        self.labels = labels
        self.resource = resource._to_dict()
        if client is None:
            client = getattr(logging.root, '_config_log_client', None)
        client = self.make_client(client, **client_kwargs, **labels)
        if isinstance(client, StreamClient):
            client.update_attachments(resource, labels, handle_name)
            self.propagate = False
        elif isinstance(client, cloud_logging.Client):  # Most likely expeected outcome.
            self.add_report_log(name)
        self.client = client  # accessing self.project may, on edge cases, set self.client
        handler = self.make_handler(handle_name, handle_level, resource, client, fmt=fmt, stream=stream, **self.labels)
        self.addHandler(handler)
        self.add_loggerDict(replace)

    def add_loggerDict(self, replace=False):
        """Emulates the effect of current logger being accessed by getLogger function and makes it available to it. """
        manager = self.manager
        existing_logger = manager.loggerDict.get(self.name, None)
        if isinstance(existing_logger, logging.PlaceHolder):
            manager._fixupChildren(existing_logger, self)
        elif existing_logger is None or replace is True:
            pass
        elif existing_logger != self:
            raise ValueError(f"A {self.name} logger already exists: {existing_logger}. Cannot replace with {self}.")
        manager.loggerDict[self.name] = self
        manager._fixupParents(self)

    @classmethod
    def get_ignore_filter(cls):
        """The 'root_high' handler may need to ignore certain loggers that are being sent to stdout by 'root_low'. """
        root_high = logging._handlers.get('root_high', None)
        if not root_high:
            raise LookupError("Could not find expected 'root_high' handler. ")
        targets = [filter for filter in root_high.filters if isinstance(filter, IgnoreFilter)]
        if len(targets) > 1:
            warnings.warn("More than one possible IgnoreFilter attached to 'root_high' handler. Using the first one. ")
        try:
            ignore_filter = targets[0]
        except IndexError:
            ignore_filter = IgnoreFilter()
            root_high.addFilter(ignore_filter)
        return ignore_filter

    @classmethod
    def get_stdout_filter(cls):
        """The filter for 'root_low' stdout handler. Allows low level logs AND to report logs recorded elsewhere. """
        root_low = logging._handlers.get('root_low', None)
        if not root_low:
            raise LookupError("Could not find expected 'root_low' handler. ")
        targets = [ea for ea in root_low.filters if isinstance(ea, LowPassFilter) and ea.title == 'stdout']
        if len(targets) > 1:
            warnings.warn("More than one possible LowPassFilter attached to 'root_low' handler. Using the first one. ")
        try:
            stdout_filter = targets[0]
        except IndexError:
            high_level = getattr(logging.root, '_config_high_level', cls.DEFAULT_HIGH_LEVEL)
            stdout_filter = LowPassFilter(name='', level=high_level, title='stdout')
            root_low.addFilter(stdout_filter)
        return stdout_filter

    @classmethod
    def add_report_log(cls, name):
        """Any level log records with this name will be sent to stdout instead of stderr when sent to root handlers. """
        stdout_filter = cls.get_stdout_filter()
        ignore_filter = cls.get_ignore_filter()
        rv = stdout_filter.allow(name)
        if isinstance(rv, str):
            rv = [rv]
        if isinstance(rv, list):
            result = [ignore_filter.add(ea) for ea in rv]
            rv = all(bool(ea) for ea in result) and len(result) > 0
        else:
            raise TypeError("Unexpected return type from adding log record name(s) to allowed for LowPassFilter. ")
        return bool(rv)

    @classmethod
    def basicConfig(cls, config=None, **kwargs):
        logging.setLoggerClass(cls)  # Causes app.logger to be a CloudLog instance.
        cred_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
        config = cls.config_as_dict(config)
        debug = kwargs.pop('debug', None) or config.get('DEBUG', None)
        testing = kwargs.pop('testing', None) or config.get('TESTING', None)
        if testing:
            return False
        base_level = cls.DEBUG_LOG_LEVEL if debug else cls.DEFAULT_LEVEL
        base_level = cls.normalize_level(kwargs.pop('level', None), base_level)
        high_level = cls.normalize_level(kwargs.pop('high_level', None), cls.DEFAULT_HIGH_LEVEL)
        if high_level < base_level:
            raise ValueError(f"The high logging level of {high_level} should be above the base level {base_level}. ")
        root_handlers = kwargs.pop('handlers', [])
        root_handlers = cls.high_low_split_handlers(base_level, high_level, root_handlers)
        log_names = kwargs.pop('log_names', [cls.APP_LOGGER_NAME])
        resource, labels, kwargs = cls.prepare_res_label(check_global=False, config=config, **kwargs)
        log_client = cls.make_client(cred_path, res_label=False, resource=resource, labels=labels, **kwargs)
        if isinstance(log_client, cloud_logging.Client):
            report_names = set()
            for name in log_names:
                handler_name = None
                if isinstance(name, tuple):
                    name, handler_name = name
                name = cls.normalize_logger_name(name)
                handler_name = cls.normalize_handler_name(handler_name or name)
                if name == cls.APP_LOGGER_NAME:
                    app_handler_name = handler_name
                report_names.add(name)
                report_names.add(handler_name)
            cls.add_report_log(report_names)
        else:  # isinstance(log_client, StreamClient):
            log_client.update_attachments(resource, labels, app_handler_name)
        kwargs['handlers'] = root_handlers
        kwargs['level'] = base_level
        try:
            logging.basicConfig(**kwargs)
            root = logging.root
            root._config_resource = resource._to_dict()
            root._config_lables = labels
            root._config_log_client = log_client
            root._config_name = name
            root._config_base_level = base_level
            root._config_high_level = high_level
        except Exception as e:
            print("********************** Unable to do basicConfig **********************")
            logging.exception(e)
            return False
        cloud_config = {'log_client': log_client, 'name': name, 'base_level': base_level, 'high_level': high_level}
        cloud_config.update({'resource': resource._to_dict(), 'labels': labels, })
        return cloud_config

    @classmethod
    def high_low_split_handlers(cls, base_level, high_level, handlers=[]):
        """Creates a split of high logs sent to stderr, low logs to stdout. Can choose some logs for always stdout. """
        low_handler = logging.StreamHandler(stdout)
        low_filter = LowPassFilter('', high_level, 'stdout')  # '' name means it applies to all logs pasing through.
        low_handler.addFilter(low_filter)
        low_handler.setLevel(base_level)
        low_handler.set_name('root_low')
        high_handler = logging.StreamHandler(stderr)
        high_handler.setLevel(high_level)
        high_handler.set_name('root_high')
        return [low_handler, high_handler, *handlers]

    @classmethod
    def getLogger(cls, name):
        return logging.getLogger(name)

    @property
    def project(self):
        """If unknown, computes & sets from labels, resource, client, environ, or created client. May set client. """
        if not getattr(self, '_project', None):
            project = self.labels.get('project', None) or self.labels.get('project_id', None)
            if not project and self.resource:
                project = self.resource.get('labels', {})
                project = project.get('project_id') or project.get('project')
            if not project and isinstance(self.client, cloud_logging.Client):
                project = self.client.project
            if not project:
                project = environ.get('GOOGLE_CLOUD_PROJECT') or environ.get('PROJECT_ID')
            if not project:
                cred_path = environ.get('GOOGLE_APPLICATION_CREDENTIALS', None)
                self.client = self.make_client(cred_path)
                project = self.client.project
            if not project:
                raise LookupError("Unable to discover the required Project id. ")
            self._project = project
        return self._project

    @classmethod
    def normalize_level(cls, level=None, default=None):
        """Returns the level value, based on the input string or integer if provided, or by using the default value. """
        if level is None and default is None:
            default = cls.DEFAULT_LEVEL
            default = default if default is not None else logging.WARNING
        level = level if level is not None else default
        clean_level = getattr(logging, '_checkLevel', _clean_level)
        level = clean_level(level)
        return level

    @classmethod
    def normalize_logger_name(cls, name=None):
        """Returns a lowercase name for a logger based on provided input or default value. """
        if not name or not isinstance(name, str):
            name = cls.DEFAULT_LOGGER_NAME
        if not name:
            raise TypeError(f"Either a logger name, or a default, string must be provided. {name} did not work. ")
        return name.lower()

    @classmethod
    def normalize_handler_name(cls, name=None):
        """Returns a lowercase name based on the given input or default value. """
        error_message = f"Either a handler name, or a default name, string must be provided. {name} did not work. "
        if name == cls.APP_LOGGER_NAME:
            name = cls.APP_HANDLER_NAME
        elif not name or not isinstance(name, str):
            name = cls.DEFAULT_HANDLER_NAME
        if not name:
            raise TypeError(error_message)
        return name.lower()

    @classmethod
    def normalize_parent(cls, parent, name):
        """Returns a logger or None, as appropriate according to the input value. """
        if not parent:
            return None
        if isinstance(parent, str):
            parent = parent.lower()
            if parent == name:
                parent = None
            else:
                parent = logging.getLogger(parent)
        elif not isinstance(parent, logging.Logger):
            raise TypeError("The 'parent' value must be a string, a logger, or None. ")
        return parent

    @classmethod
    def prepare_res_label(cls, check_global=True, **kwargs):
        """Will start with resource & labels in kwargs or found globally. Returns Resource, Labels, kwargs. """
        config = kwargs.pop('config', environ)
        res_type = kwargs.pop('res_type', cls.DEFAULT_RESOURCE_TYPE)
        resource = kwargs.pop('resource', None)
        label_overrides = kwargs.pop('labels', None)
        if check_global:
            resource = resource or getattr(logging.root, '_config_resource', {})
            label_overrides = label_overrides or getattr(logging.root, '_config_labels', {})
        else:
            resource = resource or {}
            label_overrides = label_overrides or {}
        if resource and isinstance(resource, dict):
            resource = Resource._from_dict(resource)
        labels = resource.get('labels', {})
        labels.update(kwargs)
        labels.update(label_overrides)
        if not isinstance(resource, Resource):
            resource.update(labels)
            resource = cls.make_resource(config, res_type, **resource)
        labels = resource.get('labels', {**cls.get_environment_labels(), **labels})
        labels.update(label_overrides)
        return resource, labels, kwargs

    @classmethod
    def make_client(cls, cred_or_path=None, res_label=True, check_global=True, **kwargs):
        """Creates the appropriate client, with appropriate handler for the environment, as used by other methods. """
        if isinstance(cred_or_path, (cloud_logging.Client, StreamClient)):
            return cred_or_path
        if res_label is False:
            check_global = False
        client_kwargs = {key: kwargs.pop(key) for key in cls.CLIENT_KW if key != 'project' and key in kwargs}
        if 'project' in kwargs:
            client_kwargs['project'] = kwargs['project']
        resource, labels = None, None
        if res_label:
            resource, labels, kwargs = cls.prepare_res_label(check_global, **kwargs)
        if isinstance(cred_or_path, service_account.Credentials):
            credentials = cred_or_path
        elif cred_or_path:
            credentials = service_account.Credentials.from_service_account_file(cred_or_path)
            credentials = credentials.with_scopes(cls.LOG_SCOPES)
        else:
            credentials = None
        client_kwargs.setdefault('credentials', credentials)
        log_client = None
        if cred_or_path != logging:
            try:
                log_client = cloud_logging.Client(**client_kwargs)
            except Exception as e:
                logging.exception(e)
                log_client = None
        if not log_client:
            log_client = StreamClient(**kwargs, **client_kwargs)
        if isinstance(log_client, StreamClient) and any(resource, labels):
            log_client.update_attachments(resource, labels)
        return log_client

    @classmethod
    def get_resource_fields(cls, res_type=None, **settings):
        """For a given resource type, extract the expected required fields from the kwargs passed and project_id. """
        res_type = res_type or cls.DEFAULT_RESOURCE_TYPE
        project_id = settings.pop('project_id', None) or settings.pop('project', None)
        if not project_id:
            project_id = environ.get('PROJECT_ID') or environ.get('PROJECT') or environ.get('GOOGLE_CLOUD_PROJECT')
        project_id = project_id or ''
        if not project_id:
            raise Warning("The important project id has not been found from passed settings or environment. ")
        pid = ('project_id', 'project')
        for key in cls.RESOURCE_REQUIRED_FIELDS[res_type]:
            backup_value = project_id if key in pid else ''
            if key not in settings and not backup_value:
                warnings.warn("Could not find {} for Resource {}. ".format(key, res_type))
            settings.setdefault(key, backup_value)
        return res_type, settings

    @classmethod
    def get_environment_labels(cls, config=environ):
        """Using the config dict, or environment, Returns a dict of context parameters if their values are truthy. """
        project_id = config.get('PROJECT_ID')
        project = config.get('GOOGLE_CLOUD_PROJECT') or config.get('PROJECT')
        if project and project_id and project != project_id:
            warnings.warn("The 'project' and 'project_id' are not equal: {} != {} ".format(project, project_id))
        if not any((project, project_id)):
            warnings.warn("Unable to find the critical project id setting from config. Checking environment later. ")
        project = project or project_id
        labels = {
            'gae_env': config.get('GAE_ENV'),
            'project': project,
            'project_id': project,
            'service': config.get('GAE_SERVICE'),
            'module_id': config.get('GAE_SERVICE'),
            'code_service': config.get('CODE_SERVICE'),  # Either local or GAE_SERVICE value
            'version_id': config.get('GAE_VERSION'),
            'zone': config.get('PROJECT_ZONE'),
            }
        return {k: v for k, v in labels.items() if v}

    @classmethod
    def config_as_dict(cls, config):
        """Takes a Config object or a dict. If input is None, returns os.environ. Otherwise, returns a dict. """
        if config and not isinstance(config, dict):
            config = getattr(config, '__dict__', None)
        if not config:
            config = environ
        return config

    @classmethod
    def make_resource(cls, config, res_type=None, **kwargs):
        """Creates an appropriate resource to help with logging. The 'config' can be a dict or config.Config object. """
        config = cls.config_as_dict(config)
        labels = cls.get_environment_labels(config)
        labels.update(kwargs)
        res_type, labels = cls.get_resource_fields(res_type=res_type, **labels)
        return Resource(res_type, labels)

    @classmethod
    def make_formatter(cls, fmt=DEFAULT_FORMAT, datefmt=None):
        """Creates a standard library formatter to attach to a handler. """
        if isinstance(fmt, logging.Formatter):
            return fmt
        return logging.Formatter(fmt, datefmt=datefmt)

    def log_levels_covered(self):
        """Reports what logging levels are covered by looking at attached handlers, and those on propagated parents. """
        max_level = logging.CRITICAL  # Same as logging.FATAL
        # TODO: Submit issue - self.getEffectiveLevel will falsely report a parent value if propagate is False.
        log_name = self.name
        log_construct_level = (log_name, self.level, max_level)  # Does not account for propagating effective level.
        levels = {log_construct_level, }
        names = {log_name, }
        entry = defaultdict({'min': 0, 'max': logging.CRITICAL, 'ranges': [], 'children': set(), 'parent_name': ''})
        tree = defaultdict()
        tree[log_name] = {'min': self.level, 'max': max_level, 'ranges': [], 'children': set(), 'parent_name': None}

        pass

    @classmethod
    def make_handler(cls, name=None, level=None, res=None, client=None, **kwargs):
        """The handler uses cloud logging output, or standard library stream, depending on the given client. """
        name = cls.normalize_handler_name(name)
        stream = kwargs.pop('stream', None)
        fmt = kwargs.pop('fmt', kwargs.pop('format', DEFAULT_FORMAT))
        if client is None:
            client = getattr(logging.root, '_config_log_client', None)
        if not client:
            raise TypeError("Expected a Client instance, or an already attached Client. ")
        if res is None:
            res = getattr(logging.root, '_config_resource', None)
            res = Resource._from_dict(res) if isinstance(res, dict) else res
        labels = kwargs.pop('labels', None)
        if not labels:
            labels = getattr(logging.root, '_config_labels', {})
        if isinstance(res, Resource):
            labels.update(res.get('labels', {}))
        if not labels:
            labels = cls.get_environment_labels()
        labels.update(kwargs)
        handler_kwargs = {'name': name, 'labels': labels}
        if res:
            handler_kwargs['resource'] = res
        if stream:
            handler_kwargs['stream'] = stream
        handler = CloudParamHandler(client, **handler_kwargs)  # CloudLoggingHandler, or stream if StreamClient.
        if level:
            level = cls.normalize_level(level)
            handler.setLevel(level)
        fmt = cls.make_formatter(fmt)
        handler.setFormatter(fmt)
        return handler

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def make_base_logger(cls, name=None, level=None, res=None, client=None, **kwargs):
        """Used to create a logger with a cloud handler when a CloudLog instance is not desired. """
        name = cls.normalize_logger_name(name)
        level = cls.normalize_level(level)
        logger = None
        if logging.getLoggerClass() == cls:
            try:
                logger = logging.Logger(name, level)
                cls.add_loggerDict(logger, replace=False)
            except Exception as e:
                logging.exception(e)
                logger = None
        if not logger:
            logger = logging.getLogger(name)
            logger.setLevel(level)
        fmt = kwargs.pop('fmt', kwargs.pop('format', DEFAULT_FORMAT))
        handler_name = kwargs.pop('handler_hame', name)
        handler_level = kwargs.pop('handler_level', None)
        handler = cls.make_handler(handler_name, handler_level, res, client, fmt=fmt, **kwargs)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def test_loggers(app, logger_names=list(), loggers=list(), levels=('warning', 'info', 'debug'), context=''):
        """Used for testing the log setups. """
        from pprint import pprint
        from collections import Counter

        if not app.got_first_request:
            app.try_trigger_before_first_request_functions()
        if logger_names is not None and not logger_names:
            logger_names = getattr(app, 'log_list', [])
        elif logger_names is None:
            logger_names = []
        app_loggers = [(name, getattr(app, name)) for name in logger_names if hasattr(app, name)]
        print(f"Found {len(app_loggers)} named attachments. ")
        app_loggers = [ea for ea in app_loggers if ea[1] is not None]
        print(f"Expected {len(logger_names)} and found {len(app_loggers)} named loggers. ")
        if hasattr(app, 'logger'):
            app_loggers.insert(0, ('App_Logger', app.logger))
        if loggers:
            print(f"Investigating {len(loggers)} independent loggers. ")
        if isinstance(loggers, dict):
            loggers = [(name, logger) for name, logger in loggers.items()]
        elif isinstance(loggers, (list, tuple)):
            loggers = [(num, ea) for num, ea in enumerate(loggers)]
        else:
            loggers = []
        loggers = [('root', logging.root)] + app_loggers + loggers
        print(f"Total loggers: {len(loggers)} ")
        code = app.config.get('CODE_SERVICE', 'UNKNOWN')
        print(f"\n=================== Logger Tests & Info: {code} ===================")
        found_handler_str = ''
        all_handlers = []
        for name, logger in loggers:
            adapter = None
            if isinstance(logger, logging.LoggerAdapter):
                adapter, logger = logger, logger.logger
            handlers = getattr(logger, 'handlers', [])
            all_handlers.extend(handlers)
            desc = ' ADAPTER' if adapter else ''
            if isinstance(logger, logging.PlaceHolder):
                desc += ' PLACEHOLDER'
            elif not handlers:
                desc += ' None'
            else:
                desc += ' '
            found_handler_str += f"{name}:{desc}{', '.join([str(ea) for ea in handlers])} " + '\n'
            if adapter:
                print(f"-------------------------- {name} ADAPTER Settings --------------------------")
                pprint(adapter.__dict__)
            print(f"---------------------------- {name} Logger {repr(logger)} ----------------------------")
            pprint(logger.__dict__)
            print(f'------------------------- Logger Calls: {name} -------------------------')
            for level in levels:
                if hasattr(adapter or logger, level):
                    getattr(adapter or logger, level)(' - '.join((context, name, level, code)))
                else:
                    print("{} in {}: No {} method on logger {} ".format(context, code, level, name))
        print(f"\n=================== Handler Info: found {len(all_handlers)} on tested loggers ===================")
        print(found_handler_str)
        found_clients, creds_list, resources = [], [], []
        all_handlers = [ea for ea in all_handlers if ea and ea != 'not found']
        for num, handle in enumerate(all_handlers):
            print(f"--------------------- {num}: {getattr(handle, 'name', None) or repr(handle)} ---------------------")
            pprint(handle.__dict__)
            temp_client = getattr(handle, 'client', object)
            if isinstance(temp_client, (cloud_logging.Client, StreamClient)):
                found_clients.append(temp_client)
                temp_creds = getattr(temp_client, '_credentials', None)
                if temp_creds:
                    creds_list.append(temp_creds)
            resources.append(getattr(handle, 'resource', None))
        print("\n=================== Resources found attached to the Handlers ===================")
        if hasattr(app, '_resource_test'):
            resources.append(app._resource_test)
        for res in resources:
            if hasattr(res, '_to_dict'):
                pprint(res._to_dict())
            else:
                print(f"Resource was: {res} ")
        print("\n=================== App Log Client Credentials ===================")
        log_client = getattr(app, 'log_client', None)
        app_creds = None
        if log_client is not logging:
            app_creds = getattr(log_client, '_credentials', None)
            if app_creds in creds_list:
                app_creds = None
        print(f"Currently have {len(creds_list)} creds from logger clients. ")
        creds_list = [(f"client_cred_{num}", ea) for num, ea in enumerate(set(creds_list))]
        print(f"With {len(creds_list)} unique client credentials. ")
        if log_client and not app_creds:
            print("App Log Client Creds - already included in logger clients. " + '\n')
        elif app_creds:
            print("Adding App Log Client Creds. " + '\n')
            creds_list.append(('App Log Client Creds', app_creds))
        for name, creds in creds_list:
            print(f"{name}: {creds} ")
            print(creds.expired)
            print(creds.valid)
            pprint(creds.__dict__)
            print("--------------------------------------------------")
        if not creds_list:
            print("No credentials found to report.")
        print("\n=================== Log Clients Discovered ===================")
        # for num, c in enumerate(found_clients):
        #     print(f"{num}: {c} ")
        found_count = len(found_clients)
        if log_client:
            found_clients.append(log_client)
        count = dict(Counter(found_clients))
        count['total'] = sum(val for name, val in count.items())
        count_diff = count['total'] - found_count
        pprint(count)
        message = f"Discovered {len(found_clients)} clients, "
        if count_diff:
            message += f"plus the {count_diff} expected client. "
        elif log_client:
            message += "which includes the expected client. "
        else:
            message += "and no attached log_client. "
        print(message)
        found_client = set(found_clients)
        print(f"With {len(found_client)} unique Clients. \n")
        for c in found_client:
            print(repr(c))
            print(f"Count: {count[c]} ")
            pprint(c.__dict__)
            print("--------------------------------------------------")


def setup_cloud_logging(service_account_path, base_log_level, cloud_log_level, config=None, extra=None):
    """Function to setup logging with google.cloud.logging when not on Google Cloud App Standard. """
    log_client = CloudLog.make_client(service_account_path)
    log_client.get_default_handler()
    log_client.setup_logging(log_level=base_log_level)  # log_level sets the logger, not the handler.
    # TODO: Verify - Does any modifications to the default 'python' handler from setup_logging invalidate creds?
    root_handler = logging.root.handlers[0]
    low_filter = LowPassFilter(CloudLog.APP_LOGGER_NAME, cloud_log_level, 'stdout')
    root_handler.addFilter(low_filter)
    fmt = getattr(root_handler, 'formatter', None)
    if not fmt:
        fmt = DEFAULT_FORMAT
        root_handler.setFormatter(fmt)
    resource = CloudLog.make_resource(config, CloudLog.DEFAULT_RESOURCE_TYPE)
    handler = CloudLog.make_handler(CloudLog.APP_HANDLER_NAME, cloud_log_level, resource, log_client, fmt=fmt)
    logging.root.addHandler(handler)
    if extra is None:
        extra = []
    elif isinstance(extra, str):
        extra = [extra]
    cloud_logs = [CloudLog(name, base_log_level, resource, log_client, fmt=fmt) for name in extra]
    return (log_client, *cloud_logs)


def logger_coverage(logger):
    """Determine what logging levels are covered for a given (all?) logger. """
    construct_level = logger.level
    delayed = []
    for handler in logger.handlers:
        has_external_log = isinstance((getattr(handler, 'client', None)), cloud_logging.Client)
        if has_external_log:
            delayed.append(handler)
            continue
        low = max((construct_level, handler.level))
        high = MAX_LOG_LEVEL
        ranges = []
        for filter in handler.filters:
            if isinstance(filter, LowPassFilter):
                pass
        # client = getattr(handler, 'client', None)
