"""Isolated test settings: never connect to the development database."""

from .settings import *  # noqa: F403

SECRET_KEY = "heedless-backbones-tests-only"
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
LOGGING = {}

import tempfile  # noqa: E402

# Never write family YAML into the repository from tests.
FAMILY_DATA_DIR = Path(tempfile.mkdtemp(prefix="family_data-"))  # noqa: F405
EVAL_DIR = Path(tempfile.mkdtemp(prefix="evals-"))  # noqa: F405
