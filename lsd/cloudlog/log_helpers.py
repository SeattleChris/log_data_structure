import logging
from os import environ
from google.cloud.logging.handlers import CloudLoggingHandler  # , setup_logging


def config_dict(config, add_to_dict=None):
    """Returns a dict or dict like object (os.environ) with optional updated values.
    Input:
        config: Can be a dict, or None to use os.environ, otherwise uses config.__dict__.
        add_to_dict: Either a dict, or a list/tuple of config class attributes to create a dict.
    Modifies:
        if add_to_dict is given, it may modify the input config dict or os.environ.
    Output:
        A dict, or os.environ (which has dict like methods), updated with values due to optional add_to_dict input.
    """
    if add_to_dict and not isinstance(add_to_dict, dict):  # must be an iterable of config object attributes.
        add_to_dict = {getattr(config, key, None) for key in add_to_dict}
    if config and not isinstance(config, dict):
        config = getattr(config, '__dict__', None)
    if not config:
        config = environ
    if add_to_dict:
        config.update(add_to_dict)
    return config


def _clean_level(level, named=True):
    """Used if logging._checkLevel is not available. """
    name_to_level = logging._nameToLevel
    level = _level_to_allowed_num(level, name_to_level)
    if named and level not in name_to_level.values():
        raise ValueError("The level integer does not match a named level. ")
    return level


def _level_to_allowed_num(level, name_to_level=logging._nameToLevel):
    """Returns int. Raises ValueError for invalid str (not in name_to_level) or invalid int, or TypeError if needed. """
    max_level = max(name_to_level.values())
    if isinstance(level, str):
        level = name_to_level.get(level.upper(), None)
        if level is None:
            raise ValueError("The level string was not a recognized value. ")
    elif not isinstance(level, int):
        raise TypeError("The level, or default level, must be an appropriate str or int value. ")
    if not 0 <= level <= max_level:
        raise ValueError(f"The level integer ( {level} ) was not a recognized value. Max: {max_level} ")
    return level


def standard_env(config=environ):
    """Determine code environnement, assuming environment variables 'GAE_INSTANCE' & 'GAE_ENV' are only set by GCP.
    Input:
        config: either a dict, a Config object, or (less ideally) None.
    Output:
        Boolean indicating the app is either running in 'GAE Standard' or locally.
    """
    expected = ('local', 'standard')
    if isinstance(config, dict) or config is environ:
        gae_env = config.get('GAE_ENV', None)
        gae_instance = config.get('GAE_INSTANCE', None)
    else:
        gae_env = getattr(config, 'GAE_ENV', None)
        gae_instance = getattr(config, 'GAE_INSTANCE', None)
    if config is not environ:
        gae_env = gae_env or environ.get('GAE_ENV', None)
        gae_instance = gae_instance or environ.get('GAE_INSTANCE', None)
    code_environment = 'local' if not gae_instance else gae_env
    if code_environment in expected:
        return True
    return False


def test_loggers(app, logger_names=list(), loggers=list(), levels=('warning', 'info', 'debug'), context=''):
    """Used for testing the log setups. """
    from pprint import pprint
    from collections import Counter
    from .cloud_log import StreamClient

    if not app.got_first_request:
        app.try_trigger_before_first_request_functions()
    if logger_names is None:
        logger_names = []
    elif not logger_names:
        logger_names = getattr(app, 'log_names', [])
    app_loggers = [(name, getattr(app, name)) for name in logger_names if hasattr(app, name)]
    print(f"Found {len(app_loggers)} named attachments. ")
    app_loggers = [ea for ea in app_loggers if ea[1] is not None]
    print(f"Expected {len(logger_names)} and found {len(app_loggers)} named loggers. ")
    if hasattr(app, 'logger'):
        app_loggers.insert(0, ('app.logger', app.logger))
    if loggers:
        print(f"Investigating {len(loggers)} independent loggers. ")
    if isinstance(loggers, dict):
        loggers = [(name, logger) for name, logger in loggers.items()]
        # loggers = sorted(loggers, key=lambda log: log[0])
    elif isinstance(loggers, (list, tuple)):
        loggers = [(num, ea) for num, ea in enumerate(loggers)]
    else:
        loggers = []
    adapters, placeholders, null_loggers, generate_loggers, active_loggers = [], [], [], [], []
    for ea in loggers:
        if isinstance(ea[1], logging.LoggerAdapter):
            adapters.append(ea)
        elif isinstance(ea[1], logging.PlaceHolder):
            placeholders.append(ea)
        elif all(isinstance(handler, logging.NullHandler) for handler in getattr(ea[1], 'handlers', [])):
            null_loggers.append(ea)
        elif not getattr(ea[1], 'handlers', None):
            generate_loggers.append(ea)
        else:
            active_loggers.append(ea)
    loggers = [('root', logging.root)] + app_loggers + generate_loggers
    total = len(loggers) + len(adapters) + len(placeholders) + len(null_loggers)
    print(f"Counts of node types in the tree of logging objects. Total: {total} ")
    print(f"Placeholders: {len(placeholders)} | Null Loggers: {len(null_loggers)} ")
    print(f"Active Loggers without their own handlers: {len(generate_loggers)} ")
    print(f"Active loggers: {len(loggers)} | Adapters {len(adapters)} ")
    code = app.config.get('CODE_SERVICE', 'UNKNOWN')
    print(f"\n=================== Logger Tests & Info: {code} ===================")
    found_handler_str = ''
    all_handlers = []
    for name, logger in loggers:
        adapter = None
        if isinstance(logger, logging.LoggerAdapter):
            adapter, logger = logger, logger.logger
        handlers = getattr(logger, 'handlers', [])
        all_handlers.extend(handlers)
        desc = ' ADAPTER' if adapter else ''
        if isinstance(logger, logging.PlaceHolder):
            desc += ' PLACEHOLDER'
        elif not handlers:
            desc += ' None'
        else:
            desc += ' '
        found_handler_str += f"{name}:{desc}{', '.join([str(ea) for ea in handlers])} " + '\n'
        if adapter:
            print(f"-------------------------- {name} ADAPTER Settings --------------------------")
            pprint(adapter.__dict__)
        print(f"---------------------------- {name} Logger {repr(logger)} ----------------------------")
        pprint(logger.__dict__)
        print(f'------------------------- Logger Calls: {name} -------------------------')
        for level in levels:
            if hasattr(adapter or logger, level):
                getattr(adapter or logger, level)(' - '.join((context, name, level, code)))
            else:
                print("{} in {}: No {} method on logger {} ".format(context, code, level, name))
    print(f"\n=================== Handler Info: found {len(all_handlers)} on tested loggers ===================")
    print(found_handler_str)
    found_clients, creds_list, resources = [], [], []
    all_handlers = [ea for ea in all_handlers if ea and ea != 'not found']
    for num, handle in enumerate(all_handlers):
        print(f"--------------------- {num}: {getattr(handle, 'name', None) or repr(handle)} ---------------------")
        pprint(handle.__dict__)
        temp_client = getattr(handle, 'client', object)
        if isinstance(temp_client, (GoogleClient, StreamClient)):
            found_clients.append(temp_client)
            temp_creds = getattr(temp_client, '_credentials', None)
            if temp_creds:
                creds_list.append(temp_creds)
        resources.append(getattr(handle, 'resource', None))
    print("\n=================== Resources found attached to the Handlers ===================")
    if hasattr(app, '_resource'):
        resources.append(app._resource)
    for res in resources:
        if hasattr(res, '_to_dict'):
            pprint(res._to_dict())
        else:
            print(f"Resource was: {res} ")
    print("\n=================== App Log Client Credentials ===================")
    log_client = getattr(app, 'log_client', None)
    app_creds = None
    if log_client is not logging:
        app_creds = getattr(log_client, '_credentials', None)
        if app_creds in creds_list:
            app_creds = None
    print(f"Currently have {len(creds_list)} creds from logger clients. ")
    creds_list = [(f"client_cred_{num}", ea) for num, ea in enumerate(set(creds_list))]
    print(f"With {len(creds_list)} unique client credentials. ")
    if log_client and not app_creds:
        print("App Log Client Creds - already included in logger clients. " + '\n')
    elif app_creds:
        print("Adding App Log Client Creds. " + '\n')
        creds_list.append(('App Log Client Creds', app_creds))
    for name, creds in creds_list:
        print(f"{name}: {creds} ")
        print(creds.expired)
        print(creds.valid)
        pprint(creds.__dict__)
        print("--------------------------------------------------")
    if not creds_list:
        print("No credentials found to report.")
    print("\n=================== Log Clients Discovered ===================")
    # for num, c in enumerate(found_clients):
    #     print(f"{num}: {c} ")
    found_count = len(found_clients)
    if log_client:
        found_clients.append(log_client)
    count = dict(Counter(found_clients))
    count['total'] = sum(val for name, val in count.items())
    count_diff = count['total'] - found_count
    message = f"Discovered {len(found_clients)} clients, "
    if count_diff:
        message += f"plus the {count_diff} expected client. "
    elif log_client:
        message += "which includes the expected client. "
    else:
        message += "and no attached log_client. "
    print(message)
    found_client = set(found_clients)
    print(f"With {len(found_client)} unique Clients. \n")
    for c in found_client:
        print(repr(c))
        print(f"Count: {count[c]} ")
        pprint(c.__dict__)
        print("--------------------------------------------------")


def get_named_handler(name="python", logger=logging.root):
    """Returns the CloudLoggingHandler with the matching name attached to the provided logger. """
    try:
        handle = logging._handlers.get(name)
        return handle
    except Exception as e:
        logging.exception(e)
        while logger:
            handlers = getattr(logger, 'handlers', [])
            for handle in handlers:
                if handle.name == name:
                    return handle
            logger = logger.parent
    return None


def move_handlers(source, target, log_level=None):
    """Move all the google.cloud.logging handlers from source to target logger, applying log_level if provided. """
    if not all(isinstance(logger, logging.getLoggerClass()) for logger in (source, target)):
        raise ValueError('Both source and target must be loggers. ')
    stay, move = [], []
    for handler in source.handlers:
        if isinstance(handler, CloudLoggingHandler):
            if log_level:
                handler.level = log_level
            move.append(handler)
        else:
            stay.append(handler)
    if move:
        target.handlers.extend(move)
        source.handlers.clear()
        source.handlers.extend(stay)
    return
