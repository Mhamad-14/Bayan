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
