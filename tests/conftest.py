import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Patch where the name is used (src.main's local reference), not where it's defined.
    with patch("src.main.load_all_models"):
        from src.main import app
        with TestClient(app) as c:
            yield c
