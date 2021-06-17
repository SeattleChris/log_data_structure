from flask import Flask
import logging
from .cloud_log import CloudLog, LowPassFilter, setup_cloud_logging  # , StructHandler, TempLog, CloudHandler


def create_app(config, config_overrides=dict()):
    debug = config_overrides.get('DEBUG', getattr(config, 'DEBUG', None))
    testing = config_overrides.get('TESTING', getattr(config, 'TESTING', None))
    if not testing:
        base_log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=base_log_level)  # Ensures a StreamHandler to stderr is attached.
    app = Flask(__name__)
    app.config.from_object(config)
    if config_overrides:
        app.config.update(config_overrides)
    app.debug = debug
    app.testing = testing

    @app.before_first_request
    def attach_cloud_loggers():
        build = ' First Request on Build: {} '.format(app.config.get('GAE_VERSION', 'UNKNOWN VERSION'))
        logging.info('{:*^74}'.format(build))
        cred_path = getattr(config, 'GOOGLE_APPLICATION_CREDENTIALS', None)
        root_handler = logging.root.handlers[0]
        base_level = logging.DEBUG if debug else logging.INFO
        cloud_level = logging.WARNING
        log_name = 'alert'
        log_client, alert, app_handler, c_log, res = None, None, None, None, None
        if not testing and not config.standard_env:
            log_client, alert, *skip = setup_cloud_logging(cred_path, base_level, cloud_level, config, log_name)
        elif not testing:
            try:
                log_client = CloudLog.make_client(cred_path)
            except Exception as e:
                logging.exception(e)
                log_client = logging
            # res = CloudLog.make_resource(config, fancy='I am')  # TODO: fix passing a created resource.
            alert = CloudLog(log_name, base_level, res, log_client)  # name out, propagate=True
            c_log = CloudLog('c_log', base_level, res, logging)  # stderr out, propagate=False
            app_handler = CloudLog.make_handler(CloudLog.APP_HANDLER_NAME, cloud_level, res, log_client)
            app.logger.addHandler(app_handler)  # name out, propagate=True
            low_filter = LowPassFilter(app.logger.name, cloud_level)  # Do not log at this level or higher.
            if log_client is logging:  # Hi: name out, Lo: root/stderr out; propagate=True
                root_handler.addFilter(low_filter)
            else:  # Hi: name out, Lo: application out; propagate=False
                name = app.logger.name if app.logger.name not in ('', None, app_handler.name) else 'app_low_handler'
                low_handler = CloudLog.make_handler(name, base_level, res, log_client)
                low_handler.addFilter(low_filter)
                app.logger.addHandler(low_handler)
                app.logger.propagate = False
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
        import inspect

        app.try_trigger_before_first_request_functions()
        return {
            'pprint': pprint,
            'CloudLog': CloudLog,
            'LowPassFilter': LowPassFilter,
            'StreamClient': StreamClient,
            'StreamTransport': StreamTransport,
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
