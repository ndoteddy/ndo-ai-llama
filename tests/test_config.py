from app.config import load_config


def test_defaults(monkeypatch):
    for name in ("HTTP_PORT", "MONGO_PORT", "INIT_INDEX", "FLASK_DEBUG", "DISABLE_TLS_VERIFICATION"):
        monkeypatch.delenv(name, raising=False)

    config = load_config()

    assert config.http_port == 7654
    assert config.mongo_port == 27017
    assert config.init_index is False
    assert config.debug is False
    assert config.disable_tls_verification is False


def test_env_overrides_are_cast_to_the_right_types(monkeypatch):
    monkeypatch.setenv("HTTP_PORT", "9000")
    monkeypatch.setenv("MONGO_PORT", "27018")
    monkeypatch.setenv("INIT_INDEX", "true")
    monkeypatch.setenv("FLASK_DEBUG", "true")

    config = load_config()

    assert config.http_port == 9000
    assert isinstance(config.http_port, int)
    assert config.mongo_port == 27018
    assert isinstance(config.mongo_port, int)
    assert config.init_index is True
    assert config.debug is True


def test_bool_env_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("INIT_INDEX", "TRUE")
    assert load_config().init_index is True

    monkeypatch.setenv("INIT_INDEX", "false")
    assert load_config().init_index is False
