from flask import Flask
from cloudlog import CloudLog, setup_warnings_log


def create_app(config, config_overrides=dict()):
    setup_warnings_log('log')
    log_names = [__name__, 'alert', ] or None  # 'c_log'
    add_config_dict = [] or {} or None
    log_setup = CloudLog.basicConfig(config, config_overrides, add_config_dict, log_names=log_names)
    app = Flask(__name__)
    app.config.from_object(config)
    if config_overrides:
        app.config.update(config_overrides)
    CloudLog.attach_loggers(app, config, _test_log=True, **log_setup)

    # Setup the data model. Import routes and events.
    with app.app_context():
        from . import routes  # noqa: F401
        # from . import model_db
        # model_db.init_app(app)

    @app.shell_context_processor
    def expected_shell_imports():
        from pprint import pprint
        import inspect

        rv = {'pprint': pprint, 'inspect': inspect, }
        cl_shell = CloudLog.shell_context(app)
        rv.update(cl_shell)
        return rv

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
