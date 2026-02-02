import pytest
from fastapi.testclient import TestClient

from services.classification.app.database.connections import get_db
from services.classification.app.main import app
from shared.auth import get_current_user


@pytest.fixture(scope="session")
def client():
    """Fixture to create a TestClient for the FastAPI app."""
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "test-user"}
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):

    # 🧠 Mock ML loading
    monkeypatch.setattr(
        "services.classification.app.core.model_service.ModelService.load_model_and_datasets",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        "services.classification.app.core.model_service.ModelService.get_classification_stats",
        lambda *args, **kwargs: {"accuracy": 0.91},
    )

    # 🗄️ Mock DB session
    monkeypatch.setattr("services.classification.app.database.connections.get_db", lambda: iter([None]))


@pytest.fixture(autouse=True)
def override_db():
    """Fixture to override the database dependency with a fake database."""

    def fake_db():
        class DummyDB:
            def query(self, *args, **kwargs):
                return self

            def filter_by(self, *args, **kwargs):
                return self

            def first(self):
                return None

            def add(self, *args, **kwargs):
                pass

            def commit(self):
                pass

        yield DummyDB()

    app.dependency_overrides[get_db] = fake_db
