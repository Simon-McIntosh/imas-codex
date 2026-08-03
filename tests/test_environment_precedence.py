"""Environment-loading invariants for the pytest process."""

import conftest


def test_dotenv_only_fills_unset_process_values(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "NEO4J_PASSWORD=dotenv-password\nNEO4J_USERNAME=dotenv-user\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEO4J_PASSWORD", "process-password")
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)

    conftest._load_test_environment(dotenv_path)

    assert conftest.os.environ["NEO4J_PASSWORD"] == "process-password"
    assert conftest.os.environ["NEO4J_USERNAME"] == "dotenv-user"
