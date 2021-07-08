from flask import Flask
import logging
from .cloud_log import CloudLog, LowPassFilter, StreamClient, setup_cloud_logging
from google.cloud import logging as cloud_logging


def attach_loggers(app, config=None, log_setup={}, log_names=[], test_log_setup=False):
    build = ' CloudLog setup after instantiating app on build: {} '.format(app.config.get('GAE_VERSION', 'UNKNOWN VERSION'))
    logging.info('{:*^74}'.format(build))
    testing = app.config.get('testing', False)
    debug = app.config.get('debug', False)
    test_log_setup = debug
    if isinstance(log_names, str):
        log_names = [log_names]
    cred_var = 'GOOGLE_APPLICATION_CREDENTIALS'
    cred_path = app.config.get(cred_var, None)
    if not config:
        config = app.config
    if isinstance(config, dict):
        standard_env = config.get('standard_env', None)
        cred_path = cred_path or config.get(cred_var, None)
    else:
        standard_env = getattr(config, 'standard_env', None)
        cred_path = cred_path or getattr(config, cred_var, None)
    base_level = log_setup.get('base_level', CloudLog.DEBUG_LOG_LEVEL if debug else CloudLog.DEFAULT_LEVEL)
    cloud_level = log_setup.get('high_level', CloudLog.DEFAULT_HIGH_LEVEL)
    log_client = log_setup.get('log_client', None)
    res = log_setup.get('resource', None)
    labels = log_setup.get('labels', {})
    if isinstance(res, dict):
        try:
            res = cloud_logging.Resource._from_dict(res)
        except Exception as e:
            logging.exception(e)
            labels = {**res, **labels}
            res = None
    if not res:
        res = CloudLog.make_resource(config, **labels)
        labels = {**res.labels, **labels}
    app_handler_name = CloudLog.normalize_handler_name(__name__)
    extra_loggers = []
    if testing:
        pass
    elif not standard_env:
        log_client, *extra_loggers = setup_cloud_logging(cred_path, base_level, cloud_level, config, log_names)
    elif not isinstance(log_client, (cloud_logging.Client, StreamClient)):
        log_client = CloudLog.make_client(cred_path, resource=res, labels=labels, config=config)
        report_names, app_handler_name = CloudLog.process_names([__name__, *log_names])
        app_handler_name = app_handler_name or CloudLog.APP_HANDLER_NAME
        low_filter = LowPassFilter('', cloud_level, title='stdout')  # Do not log at this level or higher.
        if isinstance(log_client, StreamClient):
            low_app_name = app_handler_name + '_low'
            low_handler = CloudLog.make_handler(low_app_name, base_level, res, log_client, stream='stdout')
            low_handler.addFilter(low_filter)
            app.logger.addHandler(low_handler)
            app.logger.propagate = False
        else:  # isinstance(log_client, cloud_logging.Client):
            CloudLog.add_report_log(report_names)
            root_handlers = logging.root.handlers
            root_handlers = CloudLog.high_low_split_handlers(base_level, cloud_level, root_handlers)
            logging.root.handlers = root_handlers
    if not testing:
        app_handler = CloudLog.make_handler(app_handler_name, cloud_level, res, log_client)
        app.logger.addHandler(app_handler)
        if not extra_loggers and log_names:
            for name in log_names:
                cur_logger = CloudLog(name, base_level, automate=True, resource=res, client=log_client)
                cur_logger.propagate = isinstance(log_client, cloud_logging.Client)
                extra_loggers.append(cur_logger)
        CloudLog.add_report_log(extra_loggers)
        if test_log_setup:
            name = 'c_log'
            c_client = StreamClient(name, res, labels)
            c_log = CloudLog(name, base_level, automate=True, resource=res, client=c_client)
            # c_log is now set for: stderr out, propagate=False
            c_log.propagate = True
            # app.c_log = c_log
            extra_loggers.append(c_log)
            log_names.append(name)
    app.log_client = log_client
    app._resource_test = res
    for logger in extra_loggers:
        setattr(app, logger.name, logger)
    app.log_list = log_names  # assumes to also check for app.logger.
    logging.debug("***************************** END PRE-REQUEST ************************************")


def create_app(config, config_overrides=dict()):
    log_names = [__name__, 'alert', 'c_log']
    log_setup = CloudLog.basicConfig(config, config_overrides, log_names=log_names[:-1])
    app = Flask(__name__)
    app.config.from_object(config)
    if config_overrides:
        app.config.update(config_overrides)
    attach_loggers(app, config, log_setup, log_names[1], True)

    # Setup the data model. Import routes and events.
    with app.app_context():
        from . import routes  # noqa: F401
        # from . import model_db
        # model_db.init_app(app)

    @app.shell_context_processor
    def expected_shell_imports():
        from pprint import pprint
        from .cloud_log import CloudLog, LowPassFilter, StreamClient, StreamTransport
        import inspect

        app.try_trigger_before_first_request_functions()
        logDict = logging.root.manager.loggerDict
        all_loggers = [logger for name, logger in logDict.items()]

        return {
            'pprint': pprint,
            'CloudLog': CloudLog,
            'LowPassFilter': LowPassFilter,
            'StreamClient': StreamClient,
            'StreamTransport': StreamTransport,
            'all_loggers': all_loggers,
            'logDict': logDict,
            'logging': logging,
            'inspect': inspect,
            }

    @app.errorhandler(500)
    def server_error(e):
        app.logger.error('================== Server Handler =====================')
        app.logger.error(e)
        if app.config.get('DEBUG'):
            return """
            An internal error occurred: <pre>{}</pre>
            See logs for full stacktrace.
            """.format(e), 500
        else:
            return "An internal error occurred. Contact admin. ", 500

    return app
