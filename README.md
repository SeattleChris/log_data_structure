# log_data_structure

**Author**: Chris L Chapman
**Version**: 0.1.0

## Overview

This package adds a number of logging features for Python Flask applications on the Google Cloud Platform (GCP). It expands a few GCP logging features, exposes some GCP logging features that are not readily available for some deployment contexts. This is most notable for projects using Google App Engine (GAE) - standard environments.

## Architecture

Designed to be deployed on Google Cloud App Engine, using:

- MySQL 5.7
- Python 3.7

Core packages required for this application:

- flask
- logging (standard library)
- google.cloud.logging_v2
- google-api-python-client

Possible packages needed (to be updated):

- google.oauth2
- flask-sqlalchemy
- Flask-Migrate
- pymysql
- google-auth-httplib2
- google-auth
- googleapis-common-protos
- requests-oauthlib
