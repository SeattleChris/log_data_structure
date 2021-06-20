from flask import Flask
import logging
from .cloud_log import CloudLog, LowPassFilter, setup_cloud_logging  # , StructHandler, TempLog, CloudHandler


def create_app(config, config_overrides=dict()):
    debug = config_overrides.get('DEBUG', getattr(config, 'DEBUG', None))
    testing = config_overrides.get('TESTING', getattr(config, 'TESTING', None))
    # if not testing:
        # base_log_level = logging.DEBUG if debug else logging.INFO
        # logging.setLoggerClass(CloudLog)  # Causes app.logger to be a CloudLog instance.
    CloudLog.basicConfig(config, debug=debug, testing=testing)  # Ensures a StreamHandler to stderr is attached.
    app = Flask(__name__)
    app.config.from_object(config)
    if config_overrides:
        app.config.update(config_overrides)
    app.debug = debug
    app.testing = testing

    def get_cloud_setup():
        """Get the content setup if CloudLog.basicConfig was successful. """
        expected = (
            '_config_resource',
            '_config_lables',
            '_config_log_client',
            '_config_name',
            '_config_base_level',
            '_config_high_level',
            '_config_app_handler',
        )
        rv = {key.lstrip('_config_'): getattr(logging.root, key, None) for key in expected}
        if any(val is None for val in rv.values()):
            rv = {}
        return rv

    @app.before_first_request
    def attach_cloud_loggers():
        build = ' First Request on Build: {} '.format(app.config.get('GAE_VERSION', 'UNKNOWN VERSION'))
        logging.info('{:*^74}'.format(build))
        cl_setup = get_cloud_setup()
        base_level = cl_setup.get('base_level', logging.DEBUG if debug else logging.INFO)
        cloud_level = cl_setup.get('high_level', logging.WARNING)
        log_name = 'alert'
        log_client, res, app_handler, alert, c_log = None, None, None, None, None
        if testing:
            pass
        elif not config.standard_env:
            cred_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
            log_client, alert, *skip = setup_cloud_logging(cred_path, base_level, cloud_level, config, log_name)
        else:
            if cl_setup:
                log_client = cl_setup.get('log_client', None)
                app_handler = cl_setup.get('app_handler', None)
                res = cl_setup.get('resource', None)
            else:
                cred_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
                try:
                    log_client = CloudLog.make_client(cred_path)
                except Exception as e:
                    logging.exception(e)
                    log_client = logging
                # res = CloudLog.make_resource(config, fancy='I am')  # TODO: fix passing a created resource.
                app_handler = CloudLog.make_handler(CloudLog.APP_HANDLER_NAME, cloud_level, res, log_client)
                low_filter = LowPassFilter(app.logger.name, cloud_level)  # Do not log at this level or higher.
                if log_client is logging:  # Hi: name out, Lo: root/stderr out; propagate=True
                    root_handler = logging.root.handlers[0]
                    root_handler.addFilter(low_filter)
                else:  # Hi: name out, Lo: application out; propagate=False
                    low_handler = CloudLog.make_handler(app.logger.name, base_level, res, log_client)
                    low_handler.addFilter(low_filter)
                    app.logger.addHandler(low_handler)
                    app.logger.propagate = False
            alert = CloudLog(log_name, base_level, res, log_client)  # name out, propagate=True
            c_log = CloudLog('c_log', base_level, res, logging)  # stderr out, propagate=False
            app.logger.addHandler(app_handler)  # name out, propagate=True
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
        all_loggers = [getattr(app, name, None) for name in app.log_list]
        all_loggers = [logging.root, app.logger] + [log for log in all_loggers if log]
        # tree = make_tree(all_loggers)

        return {
            'pprint': pprint,
            'CloudLog': CloudLog,
            'LowPassFilter': LowPassFilter,
            'StreamClient': StreamClient,
            'StreamTransport': StreamTransport,
            'all_loggers': all_loggers,
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
