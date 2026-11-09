"""***************************************************************************************
SCRIPT    : test_desempate_llm.py
OBJETIVO  : Testes do Agente de Desempate SEM precisar do Ollama instalado: sobe um
            servidor HTTP falso que imita a API do Ollama (/api/tags e /api/chat).

USO:      pytest tests/test_desempate_llm.py -v
****************************************************************************************/
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import src.agentes  # noqa: E402,F401

import _llm_local as llm  # noqa: E402
from agente_desempate_llm import AgenteDesempateLLM, normalizar_classificacao  # noqa: E402


class _FalsoOllama(BaseHTTPRequestHandler):
    modelos = ["llama3.1:latest"]
    resposta = '{"classificacao_final": "Médio", "justificativa": "tema economico geral"}'
    chamadas: list = []

    def log_message(self, *a):  # silencia o servidor
        pass

    def _json(self, obj, status=200):
        dados = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": m} for m in self.modelos]})
        else:
            self._json({}, 404)

    def do_POST(self):
        corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).chamadas.append(corpo)
        if corpo["model"] not in self.modelos:
            return self._json({"error": "model not found"}, 404)
        self._json({"message": {"role": "assistant", "content": self.resposta}})


@pytest.fixture
def servidor(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _FalsoOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"0.0.0.0:{srv.server_port}")   # 0.0.0.0 deve virar 127.0.0.1
    _FalsoOllama.modelos = ["llama3.1:latest"]
    _FalsoOllama.resposta = '{"classificacao_final": "Médio", "justificativa": "tema economico geral"}'
    _FalsoOllama.chamadas = []
    llm.limpar_cache()
    yield srv
    srv.shutdown()
    llm.limpar_cache()


def _caso():
    return pd.DataFrame([
        {"titulo": "A", "descricao": "d", "conteudo": float("nan"),
         "Avaliador 1": "Alto", "Avaliador 2": "Medio", "Avaliador 3": "Baixo"},
        {"titulo": "B", "descricao": "d", "conteudo": "c",
         "Avaliador 1": "Alto", "Avaliador 2": "Alto", "Avaliador 3": "Baixo"},
    ])


def test_host_normalizado(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
    assert llm.host_ollama() == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:9999/")
    assert llm.host_ollama() == "http://127.0.0.1:9999"


def test_escolhe_modelo_com_tag():
    assert llm.escolher_modelo("llama3.1", ["llama3.1:latest"]) == "llama3.1:latest"
    assert llm.escolher_modelo("llama3.1", ["mistral:7b"]) == "mistral:7b"
    assert llm.escolher_modelo("llama3.1", ["phi3:mini"]) is None


def test_normalizar_classificacao():
    assert normalizar_classificacao("Médio") == "Medio"
    assert normalizar_classificacao(" ALTO.") == "Alto"
    assert normalizar_classificacao("Alto|Medio|Baixo") is None


def test_servidor_fora_do_ar(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")
    llm.limpar_cache()
    d = llm.diagnosticar()
    assert not d["ok"] and "nao responde" in d["motivo"]
    res = AgenteDesempateLLM(verbose=False).resolver(
        AgenteDesempateLLM.identificar_discordancia_total(_caso()))
    assert res["classificacao_final"].tolist() == ["REVISAO_MANUAL"]
    assert "nao responde" in res["justificativa_desempate"].iloc[0]
    llm.limpar_cache()


def test_modelo_nao_baixado(servidor):
    _FalsoOllama.modelos = ["phi3:mini"]
    d = llm.diagnosticar("llama3.1")
    assert not d["ok"] and d["solucao"] == "Rode: ollama pull llama3.1"


def test_desempate_resolve_com_llm(servidor):
    agente = AgenteDesempateLLM(verbose=False)
    disc = agente.identificar_discordancia_total(_caso())
    assert len(disc) == 1                                  # so a linha sem nenhum par concordante
    res = agente.resolver(disc)
    assert res["classificacao_final"].tolist() == ["Medio"]
    corpo = _FalsoOllama.chamadas[-1]
    assert corpo["model"] == "llama3.1:latest" and corpo["format"] == "json"
    assert "Conteudo: nan" not in corpo["messages"][0]["content"]


def test_rotulo_invalido_vira_revisao_manual(servidor):
    _FalsoOllama.resposta = '{"classificacao_final": "Alto|Medio|Baixo", "justificativa": "?"}'
    res = AgenteDesempateLLM(verbose=False).resolver(
        AgenteDesempateLLM.identificar_discordancia_total(_caso()))
    assert res["classificacao_final"].iloc[0] == "REVISAO_MANUAL"
