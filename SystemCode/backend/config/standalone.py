"""Responsibilities: Select the explicit local/server policy for the end-user Compose deployment.
Implementation: Reuse production PostgreSQL, Celery and authentication settings; only loopback HTTP
installations disable HTTPS-only cookie/redirect flags. Server deployments retain production HTTPS.
Related Modules: deploy.py chooses DEPLOY_MODE; the Compose proxy shares the ASGI loopback network.
Declaration Index:
None
Variable Index:
- DEPLOY_MODE: Required local/server choice; invalid values prevent application startup.
- SECURE_SSL_REDIRECT: Disabled only for the explicit local HTTP mode.
- SESSION_COOKIE_SECURE: Disabled only for local HTTP; account authentication is still required.
- CSRF_COOKIE_SECURE: Disabled only for local HTTP; CSRF verification remains enabled.
- SECURE_HSTS_SECONDS: Zero for local HTTP; inherited production HSTS applies to server HTTPS.
"""

import os

from django.core.exceptions import ImproperlyConfigured

from .production import *  # noqa: F403

DEPLOY_MODE = os.environ["DEPLOY_MODE"]
if DEPLOY_MODE not in {"local", "server"}:
    raise ImproperlyConfigured("DEPLOY_MODE must be local or server")
if DEPLOY_MODE == "local":
    SECURE_SSL_REDIRECT = False
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SECURE = False
    SECURE_HSTS_SECONDS = 0
