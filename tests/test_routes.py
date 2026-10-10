def test_post_question_happy_path(client, fake_rag_engine):
    resp = client.post("/api/question", json={"question": "what is open5gs?", "user_id": "u1"})

    assert resp.status_code == 200
    assert resp.get_json() == {"answer": fake_rag_engine.answer}
    assert fake_rag_engine.calls == [("what is open5gs?", "u1")]


def test_post_question_missing_question(client):
    resp = client.post("/api/question", json={"user_id": "u1"})

    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_post_question_missing_user_id(client):
    resp = client.post("/api/question", json={"question": "hi"})

    assert resp.status_code == 400


def test_post_question_non_json_body(client):
    resp = client.post("/api/question", data="not json", content_type="text/plain")

    assert resp.status_code == 400


def test_post_question_empty_body(client):
    resp = client.post("/api/question")

    assert resp.status_code == 400
