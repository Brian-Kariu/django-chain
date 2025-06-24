"""
Pytest configuration for django-chain tests.
"""

import os
from typing import Any, Generator

import pytest
from django.conf import settings


def pytest_configure():
    """Configure Django settings for testing."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "examples.vanilla_django.settings")


@pytest.fixture(autouse=True)
def enable_db_access_for_all_tests(db: Any) -> None:
    """Enable database access for all tests."""


@pytest.fixture()
def test_settings() -> Generator[Any, None, None]:
    """Provide test settings."""
    original_settings = settings._wrapped.__dict__.copy()
    yield settings
    settings._wrapped.__dict__.clear()
    settings._wrapped.__dict__.update(original_settings)
