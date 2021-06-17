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
    test_dict = {'first': 'val_1', 'second': 'val_2'}
    json_dict = json.dumps(dict(encoded_format='json', **test_dict))
    local_data = {
        'page': 'Proof of Life',
        'text': 'Does this text get there?',
        'info_list': ['first_item', 'second_item', 'third_item'],
        'data_dict': test_dict,
        'json_dict': json_dict,
        'empty_list': [],
        'empty_dict': {},
        }
    if args:
        local_data['_args_'] = args
    local_data.update(kwargs)
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
    info = test_local(meaningful=False, testing='logging', mod='log', id=1)
    pprint(info)
    print("************************************************************************************")
    print(app.config.get('GAE_VERSION', 'UNKNOWN VERSION'))
    print("************************************************************************************")
    # pprint(app.config)
    CloudLog.test_loggers(app, app.log_list, context='package')
    print("--------------------------------------------------")
    return redirect(url_for('view', **info))


@app.route('/<string:mod>/<int:id>')
def view(mod, id):
    """Display some tabular content, usually from the database. """
    info = test_local(mod=mod, id_=id)
    model, template = None, 'view.html'
    data = request.data or request.form.to_dict(flat=True) or request.args or None
    # Model, model = mod_lookup(mod), None
    # model = model or Model.query.get(id)
    model = data or info
    return render_template(template, mod=mod, data=model)


# Catchall redirect route.
@app.route('/<string:page_name>/')
def render_static(page_name):
    """Catch all for undefined routes. Return the requested static page. """
    if page_name in ('favicon.ico', 'robots.txt'):
        return redirect(url_for('static', filename=page_name))
    page_name += '.html'
    return render_template(page_name)
