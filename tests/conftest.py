import pytest


@pytest.fixture(autouse=True)
def _no_login(monkeypatch):
    """API tests run without the demo login; tests/test_auth.py turns it back on."""
    monkeypatch.setenv('FORCECRM_AUTH', '0')
