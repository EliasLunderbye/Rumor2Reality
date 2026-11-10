"""Run service-independent tests by default; opt into real model checks."""
import socket

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-model-tests", action="store_true", default=False,
                     help="Run tests that load/download real embedding models")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-model-tests"):
        return
    skip = pytest.mark.skip(reason="Use --run-model-tests to load real model weights")
    for item in items:
        if "model_download" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def no_network_in_offline_tests(request, monkeypatch):
    """Fail unexpected network access rather than silently contacting services."""
    if request.node.get_closest_marker("model_download"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError("Network access is disabled in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
