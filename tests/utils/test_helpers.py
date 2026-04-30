import json
from pathlib import Path

import pytest

from src.utils.helpers import create_directories, load_json, log_experiment, save_json


def test_create_directories_cria_diretorio_unico(tmp_path):
    new_dir = tmp_path / "novo"
    create_directories(str(new_dir))
    assert new_dir.exists()


def test_create_directories_cria_multiplos_diretorios(tmp_path):
    dirs = [str(tmp_path / "a"), str(tmp_path / "b")]
    create_directories(dirs)
    assert (tmp_path / "a").exists()
    assert (tmp_path / "b").exists()


def test_create_directories_aceita_path_object(tmp_path):
    new_dir = tmp_path / "via_path"
    create_directories(new_dir)
    assert new_dir.exists()


def test_create_directories_nao_falha_se_ja_existir(tmp_path):
    create_directories(str(tmp_path))  # já existe
    assert tmp_path.exists()


def test_save_json_cria_arquivo(tmp_path):
    filepath = tmp_path / "saida.json"
    save_json({"chave": "valor"}, filepath)
    assert filepath.exists()


def test_save_json_conteudo_correto(tmp_path):
    filepath = tmp_path / "saida.json"
    save_json({"score": 0.95}, filepath)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    assert data["score"] == 0.95


def test_save_json_cria_diretorios_intermediarios(tmp_path):
    filepath = tmp_path / "sub" / "dir" / "saida.json"
    save_json({"x": 1}, filepath)
    assert filepath.exists()


def test_load_json_retorna_dicionario(tmp_path):
    filepath = tmp_path / "entrada.json"
    filepath.write_text('{"chave": "valor"}', encoding="utf-8")
    result = load_json(filepath)
    assert isinstance(result, dict)
    assert result["chave"] == "valor"


def test_load_json_levanta_erro_arquivo_ausente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "inexistente.json")


def test_log_experiment_cria_arquivo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_experiment("modelo_teste", {"accuracy": 0.9})
    assert (tmp_path / "logs" / "experiments.jsonl").exists()


def test_log_experiment_acumula_entradas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_experiment("modelo_a", {"accuracy": 0.8})
    log_experiment("modelo_b", {"accuracy": 0.9})
    lines = (tmp_path / "logs" / "experiments.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_log_experiment_conteudo_valido(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log_experiment("modelo_x", {"recall": 0.75}, params={"C": 1.0})
    line = (tmp_path / "logs" / "experiments.jsonl").read_text().strip()
    entry = json.loads(line)
    assert entry["model"] == "modelo_x"
    assert entry["metrics"]["recall"] == 0.75
    assert entry["params"]["C"] == 1.0
    assert "timestamp" in entry
