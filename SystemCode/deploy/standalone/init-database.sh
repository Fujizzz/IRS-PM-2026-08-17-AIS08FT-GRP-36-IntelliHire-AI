#!/bin/sh
# Responsibilities: Initialize a non-superuser application role and its database on an empty volume.
# Implementation: The official PostgreSQL entrypoint executes this once; psql quotes the password
# as a SQL literal and stops on errors. Existing databases are never recreated or reset.
# Related Modules: deploy.py generates APP_PASSWORD; config.standalone uses only the application role.
# Declaration Index: None.
# Variable Index: APP_PASSWORD is generated private input; POSTGRES_USER is the bootstrap administrator.
set -eu
psql --username "$POSTGRES_USER" --dbname postgres --set ON_ERROR_STOP=1 \
  --set app_password="$APP_PASSWORD" <<'SQL'
CREATE ROLE intellihire LOGIN PASSWORD :'app_password' NOSUPERUSER NOCREATEDB NOCREATEROLE;
CREATE DATABASE intellihire OWNER intellihire;
SQL
