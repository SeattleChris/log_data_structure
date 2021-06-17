from flask import json, redirect, url_for, render_template, flash, session, current_app as app
from pprint import pprint
from .cloud_log import CloudLog
# from flask import request  # abort
# from datetime import time
# from .cloud_log import CloudLog


def test_local(*args, **kwargs):
    """Useful for constructing tests, but will only work when running locally. """
    app.logger.debug("========== Home route run locally ==========")
    session['hello'] = 'Hello Session World'
    app.logger.debug(session)
    local_data = {
        'page': 'Proof of Life',
        'text': 'Does this text get there?',
        'info_list': ['first_item', 'second_item', 'third_item'],
        'data': json.dumps({'first': 'val_1', 'second': 'val_2'})
    }
    return local_data


@app.route('/')
def home():
    """Default root route """
    local_data = None
    if app.config.get('LOCAL_ENV', False):
        local_data = test_local()
        flash(local_data)
    return render_template('index.html', local_data=local_data)


@app.route('/test')
def test_route():
    """Temporary route for testing components. """
    app.logger.debug("========== Test Method for admin:  ==========")
    info = {'key1': 1, 'key2': 'two', 'key3': '3rd', 'meaningful': False, 'testing': 'logging'}
    pprint(info)
    print("************************************************************************************")
    print(app.config.get('GAE_VERSION', 'UNKNOWN VERSION'))
    print("************************************************************************************")
    # pprint(app.config)
    CloudLog.test_loggers(app, app.log_list, context='package')
    print("--------------------------------------------------")

    return redirect(url_for('admin', data=info))


@app.route('/<string:mod>/<int:id>')
def view(mod, id):
    """Display some tabular content, usually from the database. """
    info = {'mod': mod, 'id_': id, 'first': 1, 'second': 'two', 'third': '3rd', 'testing': 'logging'}
    model, template = None, 'view.html'
    # Model, model = mod_lookup(mod), None
    # model = model or Model.query.get(id)
    model = model or info
    return render_template(template, mod=mod, data=model)


# Catchall redirect route.
@app.route('/<string:page_name>/')
def render_static(page_name):
    """Catch all for undefined routes. Return the requested static page. """
    if page_name in ('favicon.ico', 'robots.txt'):
        return redirect(url_for('static', filename=page_name))
    page_name += '.html'
    return render_template(page_name)
