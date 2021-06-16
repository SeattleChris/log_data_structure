from flask import json, redirect, url_for, render_template, flash, session, current_app as app
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


# Catchall redirect route.
@app.route('/<string:page_name>/')
def render_static(page_name):
    """Catch all for undefined routes. Return the requested static page. """
    if page_name in ('favicon.ico', 'robots.txt'):
        return redirect(url_for('static', filename=page_name))
    page_name += '.html'
    return render_template(page_name)
