import logging
import warnings
from sys import stderr, stdout
from os import environ
from datetime import datetime as dt
from flask import json
from google.cloud.logging import Resource, Client as BaseClientGoogle
from google.cloud.logging.handlers import CloudLoggingHandler  # , setup_logging
from google.cloud.logging_v2.handlers.transports import BackgroundThreadTransport
from google.oauth2 import service_account
from .log_helpers import config_dict, _clean_level, standard_env

DEFAULT_FORMAT = logging._defaultFormatter  # logging.Formatter('%(levelname)s:%(name)s:%(message)s')
MAX_LOG_LEVEL = logging.CRITICAL


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
        allowed = ' '
        if len(self._allowed):
            allowed = ' and any ' + ', '.join(self._allowed)
        return '<{} only {} under {}{}>'.format(self.__class__.__name__, name, self.below_level, allowed)


class ClientResourcePropertiesMixIn:
    """Ensures a logging Client has attributes for resource, labels, handler_name - processed in that order. """

    def __init__(self, resource=None, labels=None, handler=None, **kwargs):
        self.resource = resource or {}
        self._labels = {}
        self.labels = labels or {}
        self.handler_name = self.get_handler_name(handler)
        self._kwargs = kwargs

    @property
    def labels(self):
        """Returns labels dict, finding 'project_id' if needed. Includes resource labels, keeping overridden values. """
        labels_have_valid_data = bool(self._labels.get('project_id', None))
        if not labels_have_valid_data:
            resource = getattr(self, 'resource', {})
            labels = resource.get('labels', {}) if isinstance(resource, dict) else getattr(resource, 'labels', {})
            project = labels.get('project_id', None) or labels.get('project', None)
            labels.update(self._labels)  # to maintain overridden or additional label values.
            possible_attr = ('_project', 'project', 'project_id', '_project_id')
            possible_attr = (ea for ea in possible_attr)
            while project is None:
                try:
                    project = getattr(self, next(possible_attr), None)
                except StopIteration:
                    project = ''
            labels['project_id'] = project
            self._labels.update(labels)  # If all values were None or '', then labels is not yet valid.
        return self._labels

    @labels.setter
    def labels(self, labels):
        if not isinstance(labels, dict):
            raise TypeError("Expected a dict input for labels. ")
        res_labels = getattr(self.resource, 'labels', {})
        current_res_label_values = {key: val for key, val in self._labels.items() if key in res_labels}
        self._labels = {**current_res_label_values, **labels}

    @property
    def handler(self):
        """Retrieved from a weakref created in logging. If is missing resource or labels, pass to prepare_handler. """
        if not self.handler_name:
            return None
        handler = logging._handlers.get(self.handler_name, None)
        expected_attr = ('resource', 'labels')
        if handler and not all(hasattr(handler, ea) for ea in expected_attr):
            handler = self.prepare_handler(handler)
        # elif not handler:
        #     raise KeyError(f"Unable to find '{self.handler_name}' handler. ")
        return handler

    def get_handler_name(self, handler):
        """Return name from given str (already normalized) or handler object. Pass given handler to prepare_handler. """
        if not handler:
            name = ''
        elif isinstance(handler, str):
            name = handler
        elif isinstance(handler, logging.Handler):
            name = getattr(handler, 'name', '')
            self.prepare_handler(handler)
        else:
            raise TypeError(f"Expected a str or Handler to get handler name. Failed: {handler}")
        return name

    def prepare_handler(self, handler):
        """Given an already created handler, attach properties if not present: resource, labels, project, full_name. """
        handler.resource = getattr(handler, 'resource', None) or self.resource
        handler.labels = getattr(handler, 'labels', None) or self.labels
        seed_attr = ('project', '_project', 'project_id', '_project_id')
        possible_attr = (ea for ea in [*seed_attr, *[ea + '_' for ea in seed_attr]])
        attr, project, assign_attr = None, None, False
        while project is None:
            try:
                attr = next(possible_attr)
                if assign_attr or hasattr(handler, attr):
                    assign_attr = True
                    goal_value = getattr(handler, attr, None) or self.project  # correct value if hasattr or not.
                    setattr(handler, attr, goal_value)
                    project = getattr(handler, attr, None)
                    project = project if project == goal_value else None
            except StopIteration:
                if assign_attr:
                    project, attr = self.project, None
                else:
                    project, assign_attr = None, True
                    possible_attr = (ea for ea in [*seed_attr, *[ea + '_' for ea in seed_attr]])
        external_transport = getattr(handler, 'transport', None)
        if isinstance(external_transport, StreamTransport):
            external_transport = None
        if external_transport and not hasattr(handler, 'full_name'):
            name = getattr(handler, 'name', self.handler_name)
            setattr(handler, 'full_name', f"projects/{project}/logs/{name}")
        return handler

    def update_attachments(self, resource=None, labels=None, handler=None):
        """Update in the correct order. The 'handler' can be a str; If a Handler, also passed to prepare_handler. """
        if resource and isinstance(resource, (Resource, dict)):
            self.resource = resource
        if labels and isinstance(labels, dict):
            self.labels = labels
        if handler:
            self.handler_name = self.get_handler_name(handler) or self.handler_name

    def base_kwargs_from_init(self, resource, labels, handler, **kwargs):
        """Return kwargs for base Client init. Try to determine project from init parameters. """
        BASE_CLIENT_KW = ('project', 'credentials', '_http', '_use_grpc', 'client_info', 'client_options', )
        base_kwargs = {key: kwargs.pop(key) for key in BASE_CLIENT_KW if key in kwargs}
        if not base_kwargs.get('project', None):
            res_labels = {}
            if isinstance(resource, Resource):
                res_labels = getattr(resource, 'labels', {})
            elif isinstance(resource, dict):
                res_labels = resource.get('labels', {})
            if isinstance(labels, dict):
                res_labels.update(labels)
            project = res_labels.get('project', None) or res_labels.get('project_id', None)
            if project:
                base_kwargs['project'] = project
        return base_kwargs


class GoogleClient(BaseClientGoogle, ClientResourcePropertiesMixIn):
    """Extends google.cloud.logging.Client with StreamClient signature & attr: resource, labels, handler_name. """

    def __init__(self, resource=None, labels=None, handler=None, **kwargs):
        base_kwargs = self.base_kwargs_from_init(resource, labels, handler, **kwargs)
        BaseClientGoogle.__init__(self, **base_kwargs)
        ClientResourcePropertiesMixIn.__init__(self, resource, labels, handler, **kwargs)


class StreamClient(ClientResourcePropertiesMixIn):
    """This substitute for google.cloud.logging.Client will use techniques similar to standard library logging. """

    def __init__(self, resource=None, labels=None, handler='', **kwargs):
        base_kwargs = self.base_kwargs_from_init(resource, labels, handler, **kwargs)
        for key, val in base_kwargs.items():
            setattr(self, key, val)  # This may include project.
        super().__init__(self, resource, labels, handler, **kwargs)

    def base_kwargs_from_init(self, resource, labels, handler, **kwargs):
        base_kwargs = super().base_kwargs_from_init(resource, labels, handler, **kwargs)
        for key in ('credentials', 'client_info', 'client_options'):
            base_kwargs['_' + key] = base_kwargs.pop(key, None)
        base_kwargs['_http_internal'] = base_kwargs.pop('_http', None)
        return base_kwargs

    def create_handler(self, handler_param):
        """Creates or updates a logging.Handler with the correct name and attaches the labels and resource. """
        if isinstance(handler_param, str):
            handler = logging._handlers.get(handler_param, None)
            if not handler:
                handler = logging.StreamHandler()
                handler.set_name(handler_param.lower())
        elif isinstance(handler_param, type):
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
        handler = self.prepare_handler(handler)
        return handler

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
    def handler(self):
        return self.client.logger(self.handler_name)

    @property
    def stream(self):
        return self.handler.stream

    @property
    def terminator(self):
        if not getattr(self, '_terminator', None):
            self._terminator = self.handler.terminator
        return self._terminator

    def flush(self):
        self.handler.flush()

    def handleError(self, record):
        return self.handler.handleError(record)

    @property
    def destination(self):
        if not getattr(self, '_destination', None):
            self._destination = self.stream.name.lstrip('<').rstrip('>')
        return self._destination

    def __repr__(self) -> str:
        level = logging.getLevelName(self.handler.level)
        return '<%s %s | (%s) %s>' % (self.__class__.__name__, self.destination, level, self.handler_name)


class CloudParamHandler(CloudLoggingHandler):
    """Emits log by CloudLoggingHandler technique with a valid Client, or by StreamHandler if client is None. """

    record_attr = ('_resource', '_trace', '_span_id', '_http_request', '_source_location', '_labels', '_trace_str',
                   '_span_id_str', '_http_request_str', '_source_location_str', '_labels_str', '_msg_str')

    def __init__(self, client, name='param_handler', resource=None, labels=None, stream=None, ignore=None):
        if not isinstance(client, (StreamClient, GoogleClient)):
            raise ValueError("Expected a StreamClient or other appropriate cloud logging Client. ")
        transport = StreamTransport if isinstance(client, StreamClient) else BackgroundThreadTransport
        super().__init__(client, name=name, transport=transport, resource=resource, labels=labels, stream=stream)
        # handler_attr = ('name', 'transport(client, name)', 'client', 'project_id', 'resource', 'labels')

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
    APP_LOGGER_NAME = __package__.split('.')[0]  # TODO: Update if this logging package is imported into an application.
    APP_HANDLER_NAME = 'app'
    DEFAULT_LOGGER_NAME = None
    DEFAULT_HANDLER_NAME = None
    SPLIT_LOW_NAME = 'root_low'
    SPLIT_HIGH_NAME = 'root_high'
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
    CLIENT_KW = ('project', 'handler_name', 'handler', 'credentials', 'client_info', 'client_options')
    # also: '_http', '_use_grpc'

    def __init__(self, name=None, level=logging.NOTSET, automate=False, replace=False, **kwargs):
        name = self.normalize_logger_name(name)
        if level:
            level = self.normalize_level(level)
        super().__init__(name, level=level)
        if automate:
            self.automated_structure(**kwargs)
        self.add_loggerDict(replace)

    def automated_structure(self, resource=None, client=None, check_global=True, **kwargs):
        """Typically only used for the core logers of the main code. This will add resource, labels, client, etc.
        Input:
            resource: can be None, a google.cloud.logging.Resource, or a dict translation of one.
            client: can be None, a google.cloud.logging.Client, StreamClient, credential or path to credential file.
            replace: if a Logger exists with the same name, this boolean indicates if it should be replaced.
            check_global: Boolean indicating if missing settings may be found on attributes of logging.root.
            ** Restrictions: If client is None, then a valid value must be found via check_global or cred_or_path kwarg.
            List of kwarg overrides: stream, fmt, format, parent, handler_name, handler_level, high_level,
                                    cred_or_path, res_type, labels, client_info, client_options.
            All other kwargs will be used for labels.
            If not set, the defaults for those in the list will be determined by:
            stream, fmt, format, parent: from defaults in standard library logging.
            res_type, handler_name, high_level: from CloudLog class attributes.
            handler_level: Not set (logging depends logger's level, or trickle up handlers.)
            cred_or_path: set to None. Raises ValueError if there is both a client and a cred_or_path.
            labels, (and resource if None): from check_global and constructed (or updated) from kwarg values.
            client_info, client_options: Not used if missing. Used for make_client if client not already constructed.
        Modifies current instance:
            self.client: A GoogleClient or StreamClient.
            self.resource: A dict of a google.cloud.logging.Resource.
            self.lables: A dict of key-value pairs generally used for resource, but may have additional values.
            Adds a CloudParamHandler with settings - format, stream (or external log), resource, labels, level, name.
            Adds this named instance to the LoggerDict of the logging manager (replacing existing depends on 'replace').
        """
        name = self.name
        stream = self.clean_stream(kwargs.pop('stream', None))
        fmt = kwargs.pop('fmt', kwargs.pop('format', DEFAULT_FORMAT))
        default_handle_name = self.APP_HANDLER_NAME if name == self.APP_LOGGER_NAME else self.DEFAULT_HANDLER_NAME
        default_handle_name = default_handle_name or name
        handler_name = kwargs.pop('handler_name', None) or default_handle_name
        handler_level = kwargs.pop('handler_level', None)
        high_level = self.normalize_level(kwargs.pop('high_level', None), self.DEFAULT_HIGH_LEVEL, named=False)
        parent = kwargs.pop('parent', logging.root)
        cred_or_path = kwargs.pop('cred_or_path', None)
        if client and cred_or_path:
            raise ValueError("Unsure how to prioritize the passed 'client' and 'cred_or_path' values. ")
        client = client or cred_or_path
        client_kwargs = {key: kwargs.pop(key) for key in ('client_info', 'client_options') if key in kwargs}
        resource, labels, kwargs = self.prepare_res_label(check_global=check_global, resource=resource, **kwargs)
        client_kwargs.update(resource=resource, labels=labels, handler_name=handler_name)
        if client is None and check_global:
            client = getattr(logging.root, '_config_log_client', None)
        client = self.make_client(client, **client_kwargs)
        handler = self.make_handler(handler_name, handler_level, resource, client, fmt=fmt, stream=stream, **labels)
        client.update_attachments(handler=handler)
        handler_level = handler.level
        self.addHandler(handler)
        self.client = client  # accessing self.project may, on edge cases, set self.client
        self.labels = labels
        self.resource = resource._to_dict()
        self.low_level = max([handler_level, self.level]) if handler_level else self.level
        self.high_level = high_level  # Not likely unique from self.DEFAULT_HIGH_LEVEL (but possible)
        self.parent = self.normalize_parent(parent, name)
        if isinstance(client, GoogleClient):  # Most likely expeected outcome - logs to external stream.
            self.add_report_log(self, high_level, check_global=True)
        elif isinstance(client, StreamClient):
            self.propagate = False

    @classmethod
    def basicConfig(cls, config=None, config_overrides={}, add_config=None, **kwargs):
        """Must be called before flask app is created and before logging.basicConfig (triggered by any logging).
        Input:
            config: Can be an object, dictionary or None (environ as backup). If an object, will use it's __dict__ value.
            add_config: None, or a list of attributes on the config object not already included in its __dict__.
            config_overrides: None or a dictionary of values that will overridden for the Flask app configuration.
            List of kwarg overrides: debug, testing, level, high_level, handlers, log_names, res_type, resource, labels.
            All other kwargs will be used for labels and sent to logging.basicConfig.
            If not set in kwargs, the defaults for those in the list will be determined by:
                handlers: initialized as an empty list. Allows for adding other handlers to the root logger.
                debug, testing: from config.
                level, high_level, log_names, res_type: from CloudLog class attributes.
                resource, labels: constructed from config and/or environ and kwarg values (along with res_type).
        Modifies:
            CloudLog is set as the LoggerClass for logging.
            logging.root initialized with level.
            If level < high_level (default): logging.root has high & low handlers that are set with log_names overrides.
            logging.root is given some attributes with a structure of _config_*. These are also included in the return.
            If config is a dict or becomes os.environ (given None) it may get modified if config_overrides is given.
        Returns:
            dict of settings and objects used to configure loggers after Flask app is initiated.
        """
        logging.setLoggerClass(cls)  # Causes app.logger to be a CloudLog instance.
        config = config_dict(config, add_config, config_overrides)
        cred = config.get('GOOGLE_APPLICATION_CREDENTIALS', None)
        debug = kwargs.pop('debug', config.get('DEBUG', None))
        testing = kwargs.pop('testing', config.get('TESTING', None))
        if testing:
            return False
        level = cls.DEBUG_LOG_LEVEL if debug else cls.DEFAULT_LEVEL
        level = cls.normalize_level(kwargs.pop('level', None), level)
        high_level = cls.normalize_level(kwargs.pop('high_level', None), cls.DEFAULT_HIGH_LEVEL)
        root_handlers = cls.split_std_handlers(level, high_level, kwargs.pop('handlers', []))
        log_names = kwargs.pop('log_names', [cls.APP_LOGGER_NAME])
        resource, labels, kwargs = cls.prepare_res_label(check_global=False, config=config, **kwargs)
        client = cls.make_client(cred, res_label=False, check_global=False, resource=resource, labels=labels, **kwargs)
        kwargs['handlers'] = root_handlers
        kwargs['level'] = level
        name_dict = cls.process_names(log_names)
        try:
            logging.basicConfig(**kwargs)  # logging.root, or any loggers should not have been accessed yet.
            root = logging.root
            root._config_resource = resource._to_dict()
            root._config_lables = labels
            root._config_log_client = client
            root._config_level = level
            root._config_high_level = high_level
        except Exception as e:
            print("********************** Unable to do basicConfig **********************")
            logging.exception(e)
            return False
        if isinstance(client, GoogleClient):
            report_names = set(name_dict.keys()).union(name_dict.values())
            cls.add_report_log(report_names, high_level, check_global=True)
        else:  # isinstance(log_client, StreamClient):
            pass
        cloud_config = {'level': level, 'high_level': high_level, 'name_dict': name_dict}
        cloud_config.update({'log_client': client, 'resource': resource._to_dict(), 'labels': labels, })
        return cloud_config

    @classmethod
    def attach_loggers(cls, app, config=None, _test_log=False, **log_setup):
        """Called after Flask app is initiated. Ideally 'log_setup' from cls.basicConfig, but can work otherwise.
        Input:
            app: An instantiated and configured Flask app.
            config: Best if it is the config dict or object used to configure app. Uses app.config otherwise.
            log_setup: A dict, ideally the return of CloudLog.basicConfig, can include manually created or added values.
            log_names: list of additional loggers, if any, besides the main app.logger.
            test_log_setup: For module development and possibly tests. Creates 'c_log' logger & its own StreamClient.
            Valid log_setup keys and value description:
                name_dict: The logger-handler name pairs returned from cls.process_names durning basicConfig.
                log_names: Optional overrides for name_dict. Can be given as str, list, or list of 2-tuple name pairs.
                log_client: Either a google.cloud.logging.Client, a CloudLog.StreamClient, or None to create one.
                resource: Either a google.cloud.logging.Resource, or a dict that can configure one, or None.
                labels: An optional dict to construct or override defaults in creating a Resource or applied to logger.
                high_level: Where the high-low handlers should split. Default depends on CloudLog class attributes.
                level: The logging level for the application. Default depends on CloudLog class attributes & app.debug.
            The high_low_split handlers can be overridden by setting 'high_level' equal to 'level' in log_setup.
        Modifies:
            If app.testing is True, only sets app.log_client, app._resource, app.log_names to given values.
            Otherwise it sets these to either the given or computed values along with the following -
            Creates and attaches appropriate handler to app.logger.
            Creates & attaches (app.<logger_name>) a logger for each entry of the consolidated named_dict - log_names.
            If no valid log_client, creates a google.cloud.logging.Client or CloudLog.StreamClient as appropriate.
            If creating a google.cloud.logging.Client: Ensure, if appropriate, setting high_low_split handlers.
            If creating a CloudLog.StreamClient: app.logger gets a stdout handler with filter if needed for low levels.
        Output:
            None.
        """
        app_version = app.config.get('GAE_VERSION', 'UNKNOWN VERSION')
        build = ' CloudLog setup after instantiating app on build: {} '.format(app_version)
        logging.info('{:*^74}'.format(build))
        testing = app.testing
        debug = app.debug
        _test_log = debug  # TODO: Update after creating and testing module.
        log_client = log_setup.pop('log_client', None)
        resource = log_setup.get('resource', None)
        log_names = log_setup.pop('log_names', None)
        names = cls.process_names(log_names, _names=log_setup.pop('name_dict', {}))
        log_names = [name for name in names if name != cls.APP_LOGGER_NAME]
        extra_loggers = []
        if not testing:
            cred_var = 'GOOGLE_APPLICATION_CREDENTIALS'
            cred_path = app.config.get(cred_var, None)
            if not config:
                config = app.config
            elif not cred_path:
                cred_path = config.get(cred_var) if isinstance(config, dict) else getattr(config, cred_var, None)
            level = cls.DEBUG_LOG_LEVEL if debug else cls.DEFAULT_LEVEL
            level = cls.normalize_level(log_setup.pop('level', None), level)
            high_level = cls.normalize_level(log_setup.pop('high_level', None), cls.DEFAULT_HIGH_LEVEL)
            resource, labels, log_setup = cls.prepare_res_label(config=config, **log_setup)
            if not standard_env(config):
                log_client = cls.make_client(cred_path, resource=resource, labels=labels, config=config)
                log_names, *extra_loggers = cls.non_standard_logging(log_client, level, high_level, resource, names)
            elif not isinstance(log_client, (GoogleClient, StreamClient)):
                log_client = cls.make_client(cred_path, resource=resource, labels=labels, config=config)
                log_names, *extra_loggers = cls.alt_setup_logging(app, log_client, level, high_level, resource, names)
            app_handler_name = names[cls.APP_LOGGER_NAME]
            app_handler = cls.make_handler(app_handler_name, high_level, resource, log_client)
            app.logger.addHandler(app_handler)
            cls.add_report_log(extra_loggers, high_level)
            if not extra_loggers:
                log_names, extra_loggers = cls.make_extra_loggers(names, level, log_client, resource)
        if _test_log:
            name = 'c_log'
            c_client = StreamClient(name, resource, labels)
            c_log = CloudLog(name, level, automate=True, resource=resource, client=c_client)
            # c_log is now set for: stderr out, propagate=False
            c_log.propagate = True
            extra_loggers.append(c_log)  # app.c_log = c_log
            log_names.append(name)
        app.log_client = log_client
        app._resource = resource
        for logger in extra_loggers:
            setattr(app, logger.name, logger)
        app.log_names = log_names  # assumes to also check for app.logger.
        logging.debug("***************************** END post app instantiating setup *****************************")

    @classmethod
    def make_extra_loggers(cls, names, level, client, resource, **kwargs):
        """Input names dict, plus logger parameters. Returns a logger & log names lists (excludes app.logger). """
        kwargs.update({'automate': True, 'client': client, 'resource': resource, 'level': level})
        log_names = [name for name in names if name != cls.APP_LOGGER_NAME]
        loggers = []
        for name in log_names:
            cur_logger = CloudLog(name, handler_name=names[name], **kwargs)
            cur_logger.propagate = isinstance(client, GoogleClient)
            loggers.append(cur_logger)
        return log_names, loggers

    @classmethod
    def alt_setup_logging(cls, app, log_client, level, high_level, resource, names):
        """Used for standard environment, but not using .basicConfig for pre-setup. """
        app_handler_name = names[cls.APP_LOGGER_NAME]
        report_names = set(names.keys()).union(names.values())
        if isinstance(log_client, StreamClient):
            app.logger.propagate = False
            if high_level > level:
                low_app_name = app_handler_name + '_low'
                low_handler = cls.make_handler(low_app_name, level, resource, log_client, stream='stdout')
                stdout_filter = cls.make_stdout_filter(high_level)  # Do not log at this level or higher.
                low_handler.addFilter(stdout_filter)
                app.logger.addHandler(low_handler)
        else:  # isinstance(log_client, GoogleClient):
            cls.add_report_log(report_names, high_level)
            root_handlers = logging.root.handlers
            names_root_handlers = [getattr(ea, 'name', None) for ea in root_handlers]
            needed_root_handler_names = (cls.SPLIT_LOW_NAME, cls.SPLIT_HIGH_NAME)
            if not all(ea in names_root_handlers for ea in needed_root_handler_names):
                root_handlers = cls.split_std_handlers(level, high_level, root_handlers)
                logging.root.handlers = root_handlers
        log_names, loggers = cls.make_extra_loggers(names, level, log_client, resource)
        return (log_names, *loggers)

    @classmethod
    def non_standard_logging(cls, log_client, level, high_level, resource, names):
        """Function to setup logging with google.cloud.logging when not local or on Google Cloud App Standard. """
        log_client.get_default_handler()
        log_client.setup_logging(log_level=level)  # log_level sets the logger, not the handler.
        # TODO: Verify - Does any modifications to the default 'python' handler from setup_logging invalidate creds?
        handlers = logging.root.handlers.copy()
        fmt = getattr(handlers[0], 'formatter', None) if len(handlers) else DEFAULT_FORMAT
        low_handler, high_handler, *handlers = cls.split_std_handlers(level, high_level, handlers)
        low_handler.setFormatter(fmt)
        low_filter = low_handler.filters[0]
        low_filter.allow(cls.APP_LOGGER_NAME)
        ignore_filter = high_handler.filters[0]
        ignore_filter.add(cls.APP_LOGGER_NAME)
        logging.root.handlers.clear()
        logging.root.addHandler(low_handler)
        if len(handlers):
            for handler in handlers:
                handler.addFilter(ignore_filter)
                if handler.level < high_level:
                    handler.setLevel(high_level)
                logging.root.addHandler(handler)
        else:
            high_handler.setFormatter(fmt)
            logging.root.addHandler(high_handler)
        app_handler_name = names.get(cls.APP_LOGGER_NAME, cls.APP_HANDLER_NAME)
        handler = cls.make_handler(app_handler_name, high_level, resource, log_client, fmt=fmt)
        logging.root.addHandler(handler)
        log_names, loggers = cls.make_extra_loggers(names, level, log_client, resource, fmt=fmt)
        return (log_names, *loggers)

    @classmethod
    def process_names(cls, log_names, _names=None):
        """Returns a dict of logger: handler names. Always contains app logger name and app handler name.
        Input:
            log_names: Can be a list of str, a list of 2-tuple name str pairs, a str, a dict of name str pairs, or None.
            _names: Optional dict in the same form as output. Serve as default values if not overridden from log_names.
                If _names has cls.APP_LOGGER_NAME as a key, then its value will be used to override value in return.
        Output:
            A dict with logger names as keys, and handler names as values (often identical). Always includes main app.
        """
        _names = _names or {}
        if isinstance(log_names, str):
            log_names = [log_names] if log_names not in (cls.APP_LOGGER_NAME, '') else []
        if not log_names:
            log_names = []
        elif isinstance(log_names, dict):
            log_names = [(key, val) for key, val in log_names.items()]
        elif not isinstance(log_names, list):
            raise TypeError(f"Expected a list (or dict or str or None). Bad input: {log_names} ")
        app_handler_name = _names.get(cls.APP_LOGGER_NAME, cls.normalize_handler_name(cls.APP_LOGGER_NAME))
        rv = {cls.APP_LOGGER_NAME: app_handler_name or cls.APP_HANDLER_NAME}
        for name in log_names:
            handler_name = None
            if isinstance(name, tuple):
                if len(name) != 2:
                    raise ValueError(f"Expect either single or paired names for process_names. Invalid: {name} ")
                name, handler_name = name
            name = cls.normalize_logger_name(name)
            handler_name = cls.normalize_handler_name(handler_name or name)
            rv[name] = handler_name
        _names = _names or {}
        if not isinstance(_names, dict):
            raise TypeError(f"The '_names' parameter must be falsy or a dict for process_names. Failed: {_names} ")
        rv = {**_names, **rv}  # report_names = set(rv.keys()).union(rv.values())
        return rv

    @classmethod
    def add_report_log(cls, names_or_loggers, high_level=None, low_name=None, high_name=None, check_global=False):
        """Any level record with a name from names_or_loggers is only streamed to stdout when sent to root handlers. """
        if isinstance(names_or_loggers, (str, logging.Logger)):
            names_or_loggers = [names_or_loggers]
        if not high_level and check_global:  # 0 is also not valid for high_level.
            high_level = getattr(logging.root, '_config_high_level', None)
        if not high_level:
            levels = []
            for ea in names_or_loggers:
                if isinstance(ea, str):
                    continue
                cur_high = getattr(ea, 'high_level', 0)
                if not cur_high:
                    cur_levels = (getattr(handle, 'level', 0) or 0 for handle in getattr(ea, 'handlers', []))
                    cur_levels = [ea for ea in cur_levels if ea] + [getattr(ea, 'level', 0)]
                    cur_high = max(cur_levels)
                levels.append(cur_high)
            # levels = set(getattr(a, 'high_level', a.level) for a in names_or_loggers if isinstance(a, logging.Logger))
            high_level = min(levels) if levels else None
        stdout_filter = cls.get_apply_stdout_filter(high_level, low_name)  # check_global is completed, so not passed.
        ignore_filter = cls.get_apply_ignore_filter(high_name)
        success = False
        names = stdout_filter.allow(names_or_loggers)
        if isinstance(names, list):  # allways returns list if given a list.
            success = [ignore_filter.add(name) for name in names] or [False]
            success = all(success)
        else:
            raise TypeError("Unexpected return type from adding log record name(s) to allowed for LowPassFilter. ")
        return success

    @classmethod
    def make_stdout_filter(cls, level=None, _title='stdout'):
        """Allows named logs and logs below level. Applied to a handler with stdout, typically on root logger. """
        level = level or cls.DEFAULT_HIGH_LEVEL  # Cannot be zero.
        return LowPassFilter(name='', level=level, title=_title)  # '' name means it applies to all considered logs.

    @classmethod
    def get_apply_stdout_filter(cls, high_level=None, handler_name=None, check_global=False):
        """The filter for stdout low handler. Allows low level logs AND to report logs recorded elsewhere. """
        handler_name = handler_name or cls.SPLIT_LOW_NAME
        low_handler = logging._handlers.get(handler_name, None)
        if not low_handler:
            raise LookupError(f"Could not find expected {handler_name} handler. ")
        targets = [ea for ea in low_handler.filters if isinstance(ea, LowPassFilter) and ea.title.startswith('stdout')]
        if len(targets) > -1:
            names = ', '.join(' - '.join([ea.name or '_', ea.title]) for ea in targets)
            message = f"Handler {handler_name} has multiple LowPassFilters ({names}). Using the first one. "
            warnings.warn(message)
        try:
            stdout_filter = targets[0]
        except IndexError:
            if high_level is None and check_global:
                high_level = getattr(logging.root, '_config_high_level', None)
            stdout_filter = cls.make_stdout_filter(high_level)
            low_handler.addFilter(stdout_filter)
        return stdout_filter

    @classmethod
    def get_apply_ignore_filter(cls, handler_name=None):
        """A high handler may need to ignore certain loggers that are being logged due to stdout_filter. """
        handler_name = handler_name or cls.SPLIT_HIGH_NAME
        high_handler = logging._handlers.get(handler_name, None)
        if not high_handler:
            raise LookupError(f"Could not find expected {handler_name} high handler. ")
        targets = [filter for filter in high_handler.filters if isinstance(filter, IgnoreFilter)]
        if len(targets) > 1:
            message = f"More than one possible IgnoreFilter attached to {handler_name} handler, using first one. "
            warnings.warn(message)
        try:
            ignore_filter = targets[0]
        except IndexError:
            ignore_filter = IgnoreFilter()
            high_handler.addFilter(ignore_filter)
        return ignore_filter

    @classmethod
    def setup_low_handler(cls, low_name, level, high_level):
        """Returns new or existing handler (if valid configuration, overwrites level & stream, and may add filter). """
        if not all([isinstance(level, int), isinstance(high_level, int), isinstance(low_name, str), low_name != '']):
            raise TypeError("Invalid parameters for setup_low_handler method. ")
        title, filter = None, None
        try:
            filter = cls.get_apply_stdout_filter(high_level, low_name, check_global=False)
            handler = logging._handlers[low_name]
            assert filter.level == high_level
            assert not filter.name
        except (LookupError, ReferenceError):  # create from scratch.
            title = 'stdout'
            handler, filter = None, None
        except AssertionError:  # add new extra filter to existing handler.
            title = 'stdout' if filter.name else 'stdout_' + low_name
            filter = None
        if handler:  # Checking existing handler configuration.
            try:
                transport = getattr(handler, 'transport', None)
                has_stream_transport = isinstance(transport, StreamTransport)
                only_streamhandler = issubclass(logging.StreamHandler, handler.__class__)
                assert has_stream_transport or only_streamhandler
            except AssertionError:
                handler = None
                raise KeyError(f"The {low_name} handler exists but is the wrong class or has the wrong transport. ")
        else:  # No existing handler, create one.
            handler = logging.StreamHandler(stdout)
            handler.set_name(low_name)
        if handler.stream != stdout:
            handler.setStream(stdout)
            raise UserWarning(f"The {low_name} handler stream had to be updated to stdout. ")
        if not filter:  # Either normal setup on new handler, or adding new filter to existing handler
            filter = cls.make_stdout_filter(high_level, _title=title)
            if len(title) > 6:
                filter.name = low_name
            handler.addFilter(filter)
        handler.setLevel(level)
        return handler

    @classmethod
    def setup_high_handler(cls, high_name, high_level):
        """Returns new or existing handler (if valid configuration, overwrites level & stream, and may add filter). """
        if not isinstance(high_level, int) or not isinstance(high_name, str) or high_name == '':
            raise TypeError("Invalid parameters for setup_high_handler method. ")
        high_name = high_name or cls.SPLIT_HIGH_NAME
        try:
            handler = logging._handlers[high_name]
            transport = getattr(handler, 'transport', None)
            has_stream_transport = isinstance(transport, StreamTransport)
            only_streamhandler = issubclass(logging.StreamHandler, handler.__class__)
            assert has_stream_transport or only_streamhandler
        except (LookupError, ReferenceError):  # create one
            handler = logging.StreamHandler(stderr)
            handler.set_name(high_name)
        except AssertionError:  # Not valid configuration.
            handler = None
            raise KeyError(f"The {high_name} handler exists but is the wrong class or has the wrong transport. ")
        handler.setLevel(high_level)
        if handler.stream != stderr:
            handler.setStream(stderr)
        cls.get_apply_ignore_filter(high_name)
        return handler

    @classmethod
    def split_std_handlers(cls, level, high_level, handlers=[], low_name=None, high_name=None, named_levels=True):
        """If unequal level & high_level, creates a split of high logs sent to stderr, low (or assigned) logs to stdout.
        Input:
            handlers: Optional additional handlers that will be added (usually to root) after the low & high handlers.
            low_name & high_name: Uses defaults if None, otherwise str for handler names. The handlers must be named.
            level & high_level: Required int or str valid for log level. If equal, the split loggers are not created.
            named_levels: Boolean. If True, requires the level values to be integers that have an associated level name.
        Output of list of handlers, if unequal level & high level (typical use):
            First is a stdout handler with stdout_filter. Second is a stderr handler with empty IgnoreFilter.
        Output if level is equal to high_level:
            Returns the original handler(s), or an empty list if none given.
        """
        level = cls.normalize_level(level, named=named_levels)
        high_level = cls.normalize_level(high_level, cls.DEFAULT_HIGH_LEVEL, named=named_levels)
        if level == high_level:
            return handlers
        elif level > high_level:
            raise ValueError(f"The high logging level of {high_level} should be above the base level {level}. ")
        low_name = low_name or cls.SPLIT_LOW_NAME
        high_name = high_name or cls.SPLIT_HIGH_NAME
        low_handler = cls.setup_low_handler(low_name, level, high_level)
        high_handler = cls.setup_high_handler(high_name, high_level)
        return [low_handler, high_handler, *handlers]

    @classmethod
    def getLogger(cls, name):
        return logging.getLogger(name)

    @classmethod
    def normalize_level(cls, level=None, default=None, named=True):
        """Returns the level value, based on the input string or integer if provided, or by using the default value. """
        if not any(isinstance(ea, (str, int)) for ea in (level, default)):
            raise TypeError(f"Must pass a str or int for level ({level}) or default ({default}). ")
        level = level if level is not None else default
        level = _clean_level(level, named)
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
    def clean_formatter(cls, fmt=DEFAULT_FORMAT, datefmt=None):
        """Creates a standard library formatter to attach to a handler. """
        if isinstance(fmt, logging.Formatter):
            return fmt
        return logging.Formatter(fmt, datefmt=datefmt)

    @classmethod
    def clean_stream(cls, stream):
        """If given a string of 'stdout' or 'stderr', returns appropriate sys stream. Otherwise returns input. """
        if isinstance(stream, str):
            if stream == 'stdout':
                stream = stdout
            elif stream == 'stderr':
                stream = stderr
            else:
                raise TypeError("Expecting an IO stream, if not given string of 'stdout' or 'stderr'. ")
        return stream  # Can be None. Otherwise assuming it is a valid IO stream.

    @classmethod
    def make_client(cls, cred_or_path=None, res_label=True, check_global=True, **kwargs):
        """Creates the appropriate client, with appropriate handler for the environment, as used by other methods. """
        if res_label:
            resource, labels, kwargs = cls.prepare_res_label(check_global, **kwargs)
            add_client_kwargs = dict(resource=resource, labels=labels)
        else:
            add_client_kwargs = {ea: kwargs.pop(ea) for ea in ('resource', 'labels') if ea in kwargs}
        if 'project' in kwargs:
            add_client_kwargs['project'] = kwargs['project']
        client_kwargs = {key: kwargs.pop(key) for key in cls.CLIENT_KW if key != 'project' and key in kwargs}
        client_kwargs.update(add_client_kwargs)
        if cred_or_path is None and check_global:
            cred_or_path = getattr(logging.root, '._config_log_client', None)
        if isinstance(cred_or_path, (GoogleClient, StreamClient)):
            client = cred_or_path

            return client

        if isinstance(cred_or_path, service_account.Credentials):
            credentials = cred_or_path
        elif cred_or_path:
            credentials = service_account.Credentials.from_service_account_file(cred_or_path)
            credentials = credentials.with_scopes(cls.LOG_SCOPES)
        else:
            credentials = None
        client_kwargs.setdefault('credentials', credentials)
        try:
            log_client = GoogleClient(**client_kwargs)
            assert isinstance(log_client, BaseClientGoogle)
        except Exception as e:
            logging.exception(e)
            log_client = StreamClient(**kwargs, **client_kwargs)
        return log_client

    @staticmethod
    def get_resource_fields(res_type=DEFAULT_RESOURCE_TYPE, **settings):
        """For a given resource type, extract the expected required fields from the kwargs passed and project_id. """
        env_priority_keys = ('PROJECT_ID', 'PROJECT', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT')
        project = settings.pop('project', None)
        project_id = settings.pop('project_id', None) or project
        try:
            env_project_value = (environ.get(ea, None) for ea in env_priority_keys)
            while not project_id:
                project_id = next(env_project_value)
        except StopIteration:
            project_id = ''
            raise Warning("The important project id has not been found from passed settings or environment. ")
        pid = ('project_id', 'project')
        for key in CloudLog.RESOURCE_REQUIRED_FIELDS[res_type]:
            backup_value = project_id if key in pid else ''
            if key not in settings and not backup_value:
                message = "Could not find {} for Resource {}. ".format(key, res_type)
                warnings.warn(message)
            settings.setdefault(key, backup_value)
        return res_type, settings

    @staticmethod
    def get_environment_labels(config=environ):
        """Using the config dict, or environment, Returns a dict of context parameters if their values are truthy. """
        project_id = config.get('PROJECT_ID')
        project = config.get('PROJECT') or config.get('GOOGLE_CLOUD_PROJECT') or config.get('GCLOUD_PROJECT')
        if project and project_id and project != project_id:
            message = "The 'project' and 'project_id' are not equal: {} != {} ".format(project, project_id)
            warnings.warn(message)
        if not any((project, project_id)):
            message = "Unable to find the critical project id setting from config. Checking environment later. "
            warnings.warn(message)
        labels = {
            'gae_env': config.get('GAE_ENV'),
            'project': project or project_id,
            'project_id': project_id or project,
            'service': config.get('GAE_SERVICE'),
            'module_id': config.get('GAE_SERVICE'),
            'code_service': config.get('CODE_SERVICE'),  # Either local or GAE_SERVICE value
            'version_id': config.get('GAE_VERSION'),
            'zone': config.get('PROJECT_ZONE'),
            }
        return {k: v for k, v in labels.items() if v}

    @classmethod
    def make_resource(cls, config, res_type=DEFAULT_RESOURCE_TYPE, **kwargs):
        """Creates an appropriate resource to help with logging. The 'config' can be a dict or config.Config object. """
        config = config_dict(config)
        labels = cls.get_environment_labels(config)
        label_overrides = kwargs.pop('labels', {})
        labels.update(kwargs)
        labels.update(label_overrides)
        res_type, labels = cls.get_resource_fields(res_type=res_type, **labels)
        return Resource(res_type, labels)

    @classmethod
    def prepare_res_label(cls, check_global=True, **kwargs):
        """Returns Resource, Labels, kwargs. Starts with resource & labels in kwargs (or global), creates as needed. """
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
            try:
                resource = Resource._from_dict(resource)
            except Exception as e:
                print(e)
                resource = None
        if not isinstance(resource, Resource):
            if resource:
                raise TypeError(f"The resource parameter must be a Resource, dict, or None. Did not work: {resource} ")
            res_labels = {**kwargs}
            res_labels.update(label_overrides)
            resource = cls.make_resource(config, res_type, **res_labels)
        labels = getattr(resource, 'labels', cls.get_environment_labels())
        labels.update(kwargs)
        labels.update(label_overrides)
        return resource, labels, kwargs

    @classmethod
    def make_handler(cls, name=None, level=None, res=None, client=None, **kwargs):
        """The handler uses cloud logging output, or standard library stream, depending on the given client. """
        name = cls.normalize_handler_name(name)
        stream = cls.clean_stream(kwargs.pop('stream', None))
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
            labels.update(getattr(res, 'labels', {}))
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
        fmt = cls.clean_formatter(fmt)
        handler.setFormatter(fmt)
        return handler

    @property
    def project(self):
        """If unknown, computes & sets from labels, resource, client, environ, or created client. May set client. """
        if not getattr(self, '_project', None):
            project = self.labels.get('project', None) or self.labels.get('project_id', None)
            if not project and self.resource:
                project = getattr(self.resource, 'labels', {})
                project = project.get('project_id') or project.get('project')
            if not project and isinstance(self.client, GoogleClient):
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

    def levels_covered(self, name='', level=None, external=None, ranges=True, reduced=True):
        """Returns a list of range 2-tuples for ranges covered for the named logger traversing from current logger.
        Inputs:
            Both 'name' and 'level' will default to current Logger name and level if not given.
            If external is True, only considers LogRecords sent to external (non-standard) log stream.
            If external is False, excludes LogRecords sent to external log streams.
            If external is None (default), reports range(s) sent to some output.
        Outputs, depending on input flags:
            If ranges is True, reduced is False: Only return the ranges value.
            If ranges is False, reduced is True: Only return the reduced value.
            If both are True, return a tuple of both ranges and reduced (in that order).
            If both are False, raise InputError.
        """
        from .log_helpers import levels_covered
        if not any([ranges, reduced]):
            raise ValueError("The levels_covered function should have at least one of ranges or reduced as True. ")
        if not isinstance(external, (bool, type(None))):
            raise ValueError("The 'external' parameter must be one of None, True, or False. ")
        if level is None and not name:
            level = self.level
        level = level or 0
        name = name or self.name
        result = levels_covered(self, name, level, external)
        if ranges and not reduced:
            return result[0]
        if reduced and not ranges:
            return result[1]
        return result

    @staticmethod
    def shell_context(app):
        """Triggers a first request. Returns a dictionary of key classes for shell_context_processor. """
        from .log_helpers import test_loggers
        app.try_trigger_before_first_request_functions()
        logDict = logging.root.manager.loggerDict
        return {
            'CloudLog': CloudLog,
            'LowPassFilter': LowPassFilter,
            'GoogleClient': GoogleClient,
            'StreamClient': StreamClient,
            'StreamTransport': StreamTransport,
            'test_loggers': test_loggers,
            'logDict': logDict,
            'logging': logging,
            }


def make_base_logger(name=None, level=None, res=None, client=None, **kwargs):
    """DEPRECATED. Used to create a logger with a cloud handler when a CloudLog instance is not desired. """
    name = CloudLog.normalize_logger_name(name)
    level = CloudLog.normalize_level(level)
    logger = None
    if logging.getLoggerClass() == CloudLog:
        try:
            logger = logging.Logger(name, level)
            CloudLog.add_loggerDict(logger, replace=False)
        except Exception as e:
            logging.exception(e)
            logger = None
    if not logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
    fmt = kwargs.pop('fmt', kwargs.pop('format', DEFAULT_FORMAT))
    handler_name = kwargs.pop('handler_hame', name)
    handler_level = kwargs.pop('handler_level', None)
    handler = CloudLog.make_handler(handler_name, handler_level, res, client, fmt=fmt, **kwargs)
    logger.addHandler(handler)
    return logger
