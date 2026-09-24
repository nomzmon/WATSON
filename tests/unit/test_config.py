import yaml

from watson.common.config import PROJECT_ROOT, Settings, load_configs, load_yaml


def test_settings_load_reads_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_SERVER_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test\nMODEL_SERVER_URL=http://localhost:9999\n")

    settings = Settings.load(env_file)

    assert settings.openai_api_key == "sk-test"
    assert settings.model_server_url == "http://localhost:9999"


def test_project_root_resolves_to_repo_root():
    assert (PROJECT_ROOT / "README.md").exists()


def test_load_yaml_reads_absolute_path(tmp_path):
    config_file = tmp_path / "generator.yaml"
    config_file.write_text(yaml.safe_dump({"model_id": "gemma-3-12b", "temperature": 0.7}))

    data = load_yaml(config_file)

    assert data == {"model_id": "gemma-3-12b", "temperature": 0.7}


def test_load_configs_merges_with_later_precedence(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump({"x": 1, "y": 1}))
    b.write_text(yaml.safe_dump({"y": 2}))

    merged = load_configs(a, b)

    assert merged == {"x": 1, "y": 2}