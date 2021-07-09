from flask import Flask
from .cloud_log import CloudLog


def create_app(config, config_overrides=dict()):
    log_names = [__name__, 'alert', 'c_log'] or None
    add_config_dict = [] or {} or None
    log_setup = CloudLog.basicConfig(config, config_overrides, add_config_dict, log_names=log_names[:-1])
    app = Flask(__name__)
    app.config.from_object(config)
    if config_overrides:
        app.config.update(config_overrides)
    CloudLog.attach_loggers(app, config, log_setup, log_names[1], True)

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
