# log_data_structure

**Author**: Chris L Chapman
**Version**: 0.1.0

## Overview

This package adds a number of logging features for Python Flask applications on the Google Cloud
Platform (GCP), most notably on Google App Engine (GAE) - standard environment. It exposes and expands on some GCP logging features that are not readily available for some deployment contexts.

## How to use

In your file with create_app, import CloudLog. Before instantiating your Flask app, call CloudLog.basicConfig, passing it your config (object or dict) with additional parameters if desired.

Simple Example:

```Python
from flask import Flask
from cloudlog import CloudLog


def create_app(config):
    log_setup = CloudLog.basicConfig(config)  # config should be a dict or Config instance.
    app = Flask(__name__)
    app.config.from_object(config)
    CloudLog.attach_loggers(app, config, log_setup)

    ...

    return app
```

You may want to override some default settings for basicConfig. Besides the ones listed in the following example, the additional keyword argument options are: debug, testing, level, high_level, handlers, log_names, res_type, resource, labels. If they are not set they all have default values that are probably sensible in most cases. View the CloudLog.basicConfig docstring for more information on these. Any additional keyword arguments passed will be used to construct labels to associate with the loggers, handlers, client, and LogRecords.

Example with additional parameters:

- config_overrides: if create_app can have overridden parameters, CloudLog may need to be aware of them.
- add_config: A list of class attributes to include if they aren't already included in config.__dict__.
- normal app.logger plus two additional loggers - app.alert, app.c_log

```Python
from flask import Flask
from cloudlog import CloudLog


def create_app(config, config_overrides=dict()):
    log_names = [__name__, 'alert', 'c_log']
    add_config = ['PROJECT_ID']
    log_setup = CloudLog.basicConfig(config, config_overrides, add_config=add_config log_names=log_names)
    app = Flask(__name__)
    app.config.from_object(config)
    if config_overrides:
        app.config.update(config_overrides)
    CloudLog.attach_loggers(app, config, log_setup, log_names[:1])

    ...

    return app
```

If you want to include some CLI shell context your create_app could include (within the ... sections indicated above) something like the following:

```Python
def create_app(...
    ...

    @app.shell_context_processor
    def expected_shell_imports():
        from pprint import pprint
        import inspect

        rv = {'pprint': pprint, 'inspect': inspect, }
        cloud_shell_context = CloudLog.shell_context(app)
        rv.update(cloud_shell_context)
        return rv

    ...
    return app
```

## Architecture

Designed to be deployed on Google Cloud App Engine, using:

- Python 3.7

Core packages required for this application:

- flask
- logging (standard library)
- google.cloud.logging_v2
- google-api-python-client

Possible packages needed (to be updated):

- google.oauth2
- google-auth-httplib2
- google-auth
- googleapis-common-protos
- requests-oauthlib
