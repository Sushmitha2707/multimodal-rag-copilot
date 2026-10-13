import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import copilot.api as api  # noqa: E402


class FakePipeline:
    class store:  # noqa: N801
        @staticmethod
        def add(chunks):
            return len(chunks)

    def answer(self, question, provider=None, k=None):
        if provider == "bogus":
            raise ValueError("Unknown provider 'bogus'")
        return {
            "answer": "24 days [1]",
            "sources": [
                {
                    "n": 1,
                    "source": "leave_policy.md",
                    "modality": "text",
                    "locator": "full",
                    "score": 0.1,
                    "snippet": "x",
                }
            ],
            "context": "[1] ctx",
            "latency_s": 0.01,
            "provider": provider,
        }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_DB", str(tmp_path / "fb.sqlite"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr(api, "_pipeline", FakePipeline())
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_logs_interaction_and_hides_context(client):
    r = client.post("/ask", json={"question": "How many leave days?", "provider": "gemini"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("24") and body["interaction_id"] >= 1 and "context" not in body


def test_feedback_flow_and_validation(client):
    iid = client.post("/ask", json={"question": "How many leave days?"}).json()["interaction_id"]
    ok = client.post("/feedback", json={"interaction_id": iid, "rating": -1, "corrected_answer": "24 days."})
    assert ok.status_code == 200
    assert client.post("/feedback", json={"interaction_id": iid, "rating": 7}).status_code == 400
    assert client.post("/feedback", json={"interaction_id": 9999, "rating": 1}).status_code == 404


def test_unknown_provider_is_400(client):
    assert client.post("/ask", json={"question": "anything?", "provider": "bogus"}).status_code == 400


def test_api_key_enforced_when_set(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    assert client.post("/ask", json={"question": "hello there"}).status_code == 401
    assert client.post("/ask", json={"question": "hello there"}, headers={"x-api-key": "secret"}).status_code == 200


def test_ingest_rejects_unsupported_extension(client):
    r = client.post("/ingest", files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 200 and r.json()["chunks_indexed"] == 0 and "evil.exe" in r.json()["errors"]


def test_graph_mode_uses_graph_engine_and_returns_trace(client, monkeypatch):
    class FakeGraph(FakePipeline):
        def answer(self, question, provider=None, k=None):
            out = super().answer(question, provider, k)
            out["trace"] = ["retrieve: q", "grade: yes", "generate"]
            return out

    monkeypatch.setattr(api, "_graph", FakeGraph())
    body = client.post("/ask", json={"question": "How many leave days?", "mode": "graph"}).json()
    assert body["trace"][-1] == "generate"
    assert client.post("/ask", json={"question": "How many leave days?", "mode": "bogus"}).status_code == 422
