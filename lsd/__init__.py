from flask import Flask
import logging
from .cloud_log import CloudLog, LowPassFilter, StreamClient, setup_cloud_logging
from google.cloud import logging as cloud_logging


def create_app(config, config_overrides=dict()):
    debug = config_overrides.get('DEBUG', getattr(config, 'DEBUG', None))
    testing = config_overrides.get('TESTING', getattr(config, 'TESTING', None))
    log_names = [__name__, 'alert', 'c_log']
    log_setup = CloudLog.basicConfig(config, debug=debug, testing=testing, log_names=log_names)
    app = Flask(__name__)
    app.config.from_object(config)
    if config_overrides:
        app.config.update(config_overrides)
    app.debug = debug
    app.testing = testing
    app._log_setup = log_setup
    app.log_list = log_names[1:]

    @app.before_first_request
    def attach_cloud_loggers():
        build = ' First Request on Build: {} '.format(app.config.get('GAE_VERSION', 'UNKNOWN VERSION'))
        logging.info('{:*^74}'.format(build))
        cred_var = 'GOOGLE_APPLICATION_CREDENTIALS'
        cred_path = app.config.get(cred_var, getattr(config, cred_var, None))
        log_setup = getattr(app, '_log_setup', {})
        base_level = log_setup.get('base_level', logging.DEBUG if debug else logging.INFO)
        cloud_level = log_setup.get('high_level', logging.WARNING)
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
        log_name = 'alert'
        app_handler, alert, c_log = None, None, None
        if testing:
            pass
        elif not config.standard_env:
            log_client, alert, *skip = setup_cloud_logging(cred_path, base_level, cloud_level, config, log_name)
        elif not isinstance(log_client, (cloud_logging.Client, StreamClient)):
            log_client = CloudLog.make_client(cred_path, resource=res, labels=labels, config=config)
            # log_names = [__name__] + getattr(app, 'log_list', ['alert', 'c_log'])
            log_names = [__name__, 'alert']
            report_names, app_handler_name = CloudLog.process_names(log_names)
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
            alert = CloudLog(log_name, base_level, automate=True, resource=res, client=log_client)
            alert.propagate = isinstance(log_client, cloud_logging.Client)
            c_client = StreamClient('c_log', res, labels)
            c_log = CloudLog('c_log', base_level, automate=True, resource=res, client=c_client)  # stderr out, propagate=False
            c_log.propagate = True
            CloudLog.add_report_log(alert)
        app.log_client = log_client
        app._resource_test = res
        app.alert = alert
        app.c_log = c_log
        app.log_list = ['alert', 'c_log']  # assumes to also check for app.logger.
        logging.debug("***************************** END PRE-REQUEST ************************************")

    # Setup the data model. Import routes and events.
    with app.app_context():
        from . import routes  # noqa: F401
        # from . import model_db
        # model_db.init_app(app)

    @app.shell_context_processor
    def expected_shell_imports():
        from pprint import pprint
        from .cloud_log import CloudLog, LowPassFilter, StreamClient, StreamTransport
        # from .node_tree import LogNode, RootLogNode, handler_ranges, get_tree, make_tree, all_nodes
        import inspect

        app.try_trigger_before_first_request_functions()
        logDict = logging.root.manager.loggerDict
        all_loggers = [logger for name, logger in logDict.items()]
        # all_loggers = [getattr(app, name, None) for name in app.log_list]
        # all_loggers = [logging.root, app.logger] + [log for log in all_loggers if log]
        # tree = make_tree(all_loggers)

        return {
            'pprint': pprint,
            'CloudLog': CloudLog,
            'LowPassFilter': LowPassFilter,
            'StreamClient': StreamClient,
            'StreamTransport': StreamTransport,
            'all_loggers': all_loggers,
            'logDict': logDict,
            # 'LogNode': LogNode,
            # 'RootLogNode': RootLogNode,
            # 'handler_ranges': handler_ranges,
            # 'get_tree': get_tree,
            # 'make_tree': make_tree,
            # 'tree': tree,
            # 'all_nodes': all_nodes,
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
