# sample-django

This serves as a structure for developing, or investigating, stand alone Django app packages. This provides useful utilities and a simple isolated Django project environment. This also serves as a place to collect tools for general Django project development. This includes a variety of code snippets, utilities, content for mock-up and/or initial templates, and other resources that can be useful for both initial and ongoing Django projects. While these pieces do not warrant their own installable packages, the function snippets or resources that are useful for any given project can be copied and adapted as appropriate.

## Utilities and Resources for Django Projects

The aspects that are primarily concerning general Django Project development are
[documented in this collection](project_tools.md).

## Making an Installable Django App as a Package Published to PyPI

An installable Django app needs to be made into a package. This generally means it should be made into an egg, wheel, or source distribution. These are built with setup tools.

Overview Steps:

* Create an app with the standard django-admin commands. Develop as desired & confirm it works.
* Make sure there are tests for the app, and they work, before the following steps to make it a stand alone app.
* Move the app code to a new directory at the root of the repo. This directory name can match the app name.
* Do a global search for APPNAME, exclude `APPNAME_DIRECTORY, README.md, web/` and replace it with the stand alone app name.
* Confirm project changes:
  * settings should include the app in the installed_apps list
  * urls should include the app urls.
* Confirm utility files changes: `makemigrations.py`, `load_tests.py`, `boot_django.py`
* Modify `setup.cfg` with the appropriate name and url, and possibly other settings.
* Install the local package:
  * `pipenv install -e .`
  * TODO: UPDATE PIP COMMAND LINE HERE
* Confirm the project works, recognizing the app from the installed packages.
* Create a repo for the standalone Django app. Move all app code and expected typical support files.
  * APP_DIRECTORY
  * APPNAME.egg-info
  * cp path/cur/{README.rst,setup.cfg,setup.py,load_tests.py,tox.ini,boot_django.py,pyroject.toml,} path/end
* Update `setup.cfg` with the new repo for url, and any other changes needed.
* Confirm all tests and functionality as expected and documented.
* Publish to PyPI (see section below).

Expected typical support files:

* README.rst - general information about the package (README.md is possible with extra parameters).
* setup.cfg - has metadata and options sections for clearly structured build and description of the package.
* setup.py - shim to execute the setup.cfg file.
* pyproject.toml - See [Brett Cannon's article](https://snarky.ca/what-the-heck-is-pyproject-toml/)
* tox.ini - defines which combinations of environments to test.
* [app directory] - holding the actual Django app code as well as tests.

See the sample docs in this repository.

### Utility Files

The following files may need additional modifications for a given stand alone Django app we are making or testing.

* boot_django.py - sets up & configure Django settings as needed for the stand alone Django app.
* djangoshell.py - start a shell that is aware of our configured Django project including the stand alone app.
* makemigrations.py - creates the migration files needed for the stand alone Django app models.
* migrate.py - performs table migrations so the database is setup as needed.
* load_tests.py - can run all or some tests.Used by package manager to get its test suite, and by tox to run its tests.

### Testing various versions with tox

The `tox` tool can help us see if our package works on a variety of environments, such as different Python versions.
For example, to test Python 3.6 & 3.7, combined with testing Django 2.2 & 3.0 (four environments total) use:

```tox.ini
[tox]
envlist = py{36,37}-django220, py{36,37}-django300

[testenv]
deps =
    django220: Django>=2.2,<3
    django300: Django>=3
commands=
    python setup.py test
```

### Publishing as a package hosted on PyPI

PyPI expects an egg, wheel, or source distribution.

We can use [Twine](https://twine.readthedocs.io/en/latest/) to build and upload (there are other options).

```Shell
python -m pip install -U wheel twine setuptools
python setup.py sdist
python setup.py bdist_wheel
twine upload dist/*
```
