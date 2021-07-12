# log_data_structure

**Author**: Chris L Chapman
**Version**: 0.1.0

## Overview

This package adds a number of logging features for Python Flask applications on the Google Cloud
Platform (GCP), most notably on Google App Engine (GAE) - standard environment. It exposes and expands on some GCP logging features that are not readily available for some deployment contexts.

This sets the root logger to report higher level LogRecords to stderr, and lower level LogRecords to stdout. This can be overridden by setting the 'high_level' to equal 'level' in passed parameters, or the class attributes of 'DEFAULT_LEVEL' and 'DEFAULT_HIGH_LEVEL'. If a logger sends to an external stream, then by default all of it's LogRecords will also be sent to stdout and not to stderr. The intent is to allow watching the stdout and stderr streams and not miss anything, but reviewing log reports can be simplified by filtering to stderr and individual external logs as desired.

## How to use

### Using CloudLog.basicConfig (preferred method)

Must call CloudLog.basicConfig called before instantiating, or creating, the Flask app. Of course this can be within the 'create_app' function or otherwise called before `app = Flask(__name__)`.

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

- config_overrides: if create_app can override config settings, CloudLog may need to be aware of them.
- add_config: A list of class attributes to include if they aren't already included in config.__dict__.
- Add two loggers (app.alert and app.x_log) in addition to the default app.logger.

```Python
from flask import Flask
from cloudlog import CloudLog


def create_app(config, config_overrides=dict()):
    log_names = [__name__, 'alert', 'x_log']
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

### Without CloudLog.basicConfig

Attempts to do most of the work using the normal technique (calling CloudLog.basicConfig before creating the Flask app). However, CloudLog will not be set as the logging LoggerClass. This technique is experimental and may not be as reliable as the other technique.

Simple Example:

```Python
from flask import Flask
from cloudlog import CloudLog


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    CloudLog.attach_loggers(app, config)

    ...

    return app
```

Can also pass manual values in leu of those settings setup through the CloudLog.basicConfig technique. This could be something like replacing the single `CloudLog.attach_loggers(app, config)` with the following example:

```Python
  ...

  labels = {'user_setting': 'user_value'}
  log_setup = {'level': 10, 'high_level': 30, 'labels': labels}
  log_names = ['alert']
  CloudLog.attach_loggers(app, config, log_setup, log_names)
  ...
```

See CloudLog.attach_loggers docstring for more information on appropriate log_setup keys and values.

### Other Setup Features

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
