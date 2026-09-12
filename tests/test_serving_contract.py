"""Lab 7/capstone API contract smoke tests."""
from fastapi.testclient import TestClient
from bayan.serving.api import app

client = TestClient(app)


def test_health_endpoint_exists():
    response = client.get("/health")
    assert response.status_code == 200


def test_classify_endpoint_exists():
    response = client.post("/v1/classify", json={"text": "الخدمة ممتازة"})
    # During the starter phase this may fail until Lab 7 is implemented;
    # after Lab 7 it must become a successful contract.
    assert response.status_code == 200



def test_batch_classify_endpoint_exists():
    response = client.post(
        "/v1/classify:batch",
        json={"texts": ["الخدمة ممتازة", "The road needs maintenance"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert all("label" in item for item in data["results"])


def test_search_endpoint_contract(monkeypatch):
    class DummySearch:
        def search(self, query, k=5, candidates=50, min_score=0.25):
            return [{"case_id": "CASE-TEST", "topic": "roads", "score": 0.9}]

    monkeypatch.setattr("bayan.serving.api.get_searcher", lambda: DummySearch())

    response = client.post(
        "/v1/search",
        json={"query": "road maintenance", "k": 3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "road maintenance"
    assert data["results"][0]["case_id"] == "CASE-TEST"



def test_entities_endpoint_contract(monkeypatch):
    monkeypatch.setattr(
        "bayan.serving.api.get_ner_pipeline",
        lambda: lambda text: [
            {
                "entity_group": "LOCATION",
                "word": "الرياض",
                "score": 0.99,
                "start": 0,
                "end": 6,
            }
        ],
    )

    response = client.post("/v1/entities", json={"text": "الرياض"})
    assert response.status_code == 200
    assert response.json()["entities"][0]["entity"] == "LOCATION"


def test_analyse_endpoint_contract(monkeypatch):
    monkeypatch.setattr(
        "bayan.serving.api.PREDICTOR.predict",
        lambda text: {"label": "roads", "confidence": 0.9},
    )

    monkeypatch.setattr(
        "bayan.serving.api.get_ner_pipeline",
        lambda: lambda text: [
            {
                "entity_group": "LOCATION",
                "word": "الرياض",
                "score": 0.99,
                "start": 0,
                "end": 6,
            }
        ],
    )

    class DummySearch:
        def search(self, query, k=3, candidates=50, min_score=0.25):
            return [{"case_id": "CASE-TEST", "topic": "roads", "score": 0.9}]

    monkeypatch.setattr(
        "bayan.serving.api.get_searcher",
        lambda: DummySearch(),
    )

    response = client.post(
        "/v1/analyse",
        json={"text": "road maintenance", "k": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["classification"]["label"] == "roads"
    assert data["entities"][0]["entity"] == "LOCATION"
    assert data["similar_cases"][0]["case_id"] == "CASE-TEST"
