from .cloud_log import CloudLog, CloudParamHandler, IgnoreFilter, LowPassFilter, StreamClient, GoogleClient
from .log_helpers import get_named_handler, move_handlers, setup_warnings_log
from .range_helpers import levels_covered

__modulename__ = 'cloudlog'
__version__ = '0.1.0'
__author__ = 'Chris L Chapman'
__license__ = 'MIT'
__repository__ = 'https://github.com/seattlechris/cloudlog'
__description__ = """Expanded logging features on GCP - GAE with google.cloud.logging
[Google Cloud Platform - Google App Engine]."""

# __all_ = [
#     'CloudLog',
#     'CloudParamHandler',
#     'IgnoreFilter',
#     'LowPassFilter',
#     'StreamClient',
#     'GoogleClient',
#     'get_named_handler',
#     'move_handlers',
#     'levels_covered',
# ]
