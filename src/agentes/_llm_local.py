"""
UTILITARIO COMPARTILHADO - CHAMADA A LLM LOCAL VIA OLLAMA
=============================================================
O Agente de Desempate (e, opcionalmente, os Agentes 4 e 5) usa uma LLM
rodando LOCALMENTE via Ollama (https://ollama.com), custo zero.

CORRECAO (OUT/2026): a versao anterior dependia da biblioteca Python
`ollama`, que estava comentada no requirements.txt e nao existia no venv.
O `import` falhava em silencio e llm_disponivel() devolvia False mesmo com
o Ollama rodando - todos os casos iam para REVISAO_MANUAL. Agora o modulo
fala direto com a API HTTP do Ollama usando so a biblioteca padrao
(urllib), e o diagnostico diz exatamente o que esta faltando.

PRE-REQUISITO (uma vez so):
    1. Instalar o Ollama:  https://ollama.com/download
    2. Baixar um modelo:   ollama pull llama3.1
       (mais leves para CPU fraca: llama3.2 / mistral)
    3. Deixar o Ollama aberto (servidor local na porta 11434)

VARIAVEIS DE AMBIENTE (opcionais):
    OLLAMA_HOST        endereco do servidor (padrao http://127.0.0.1:11434).
                       "0.0.0.0" e "localhost" viram 127.0.0.1 no cliente.
    DESEMPATE_MODELO   modelo a usar (padrao llama3.1). Se nao estiver
                       baixado, usa o primeiro disponivel de MODELOS_ALTERNATIVOS.
    OLLAMA_TIMEOUT     segundos por chamada (padrao 300; a 1a chamada carrega
                       o modelo na memoria e pode demorar).

DIAGNOSTICO:
    python src/agentes/_llm_local.py
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from functools import lru_cache

MODELO_PADRAO = os.environ.get("DESEMPATE_MODELO", "llama3.1")
MODELOS_ALTERNATIVOS = ["llama3.1", "llama3.2", "llama3", "mistral", "qwen2.5", "gemma2"]
TIMEOUT_CHAMADA = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
TIMEOUT_PING = 5.0


# ----------------------------------------------------------------------
# Endereco do servidor
# ----------------------------------------------------------------------
def host_ollama() -> str:
    """Normaliza OLLAMA_HOST para uma URL que o cliente consegue acessar."""
    bruto = (os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434").strip().rstrip("/")
    if "://" not in bruto:
        bruto = "http://" + bruto
    esquema, resto = bruto.split("://", 1)
    hostporta = resto.split("/", 1)[0]
    if ":" in hostporta.rsplit("]", 1)[-1]:
        host, porta = hostporta.rsplit(":", 1)
    else:
        host, porta = hostporta, "11434"
    # 0.0.0.0 / "localhost" sao enderecos de ESCUTA do servidor; no Windows o
    # cliente falha ao conectar em 0.0.0.0 e "localhost" pode resolver para
    # IPv6 (::1) enquanto o Ollama escuta so em IPv4.
    if host in ("0.0.0.0", "", "localhost", "[::]"):
        host = "127.0.0.1"
    return f"{esquema}://{host}:{porta}"


def _requisicao(caminho: str, corpo: dict | None = None, timeout: float = TIMEOUT_PING) -> dict:
    url = host_ollama() + caminho
    dados = json.dumps(corpo).encode("utf-8") if corpo is not None else None
    req = urllib.request.Request(url, data=dados, method="POST" if dados else "GET",
                                 headers={"Content-Type": "application/json"})
    # ignora proxy do sistema para o servidor local
    abridor = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with abridor.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ----------------------------------------------------------------------
# Modelos instalados
# ----------------------------------------------------------------------
def listar_modelos() -> list[str]:
    """Nomes dos modelos baixados no Ollama (ex.: ['llama3.1:latest'])."""
    return [m.get("name") or m.get("model", "") for m in _requisicao("/api/tags").get("models", [])]


def _base(nome: str) -> str:
    return nome.split(":", 1)[0].lower()


def escolher_modelo(preferido: str = MODELO_PADRAO, instalados: list[str] | None = None) -> str | None:
    """
    Casa o modelo pedido com o que esta instalado. 'llama3.1' casa com
    'llama3.1:latest' ou 'llama3.1:8b'. Se o preferido nao estiver
    baixado, usa o primeiro de MODELOS_ALTERNATIVOS que estiver.
    """
    instalados = listar_modelos() if instalados is None else instalados
    if not instalados:
        return None
    for nome in instalados:                       # nome exato (com tag)
        if nome.lower() == preferido.lower():
            return nome
    for nome in instalados:                       # mesma familia, qualquer tag
        if _base(nome) == _base(preferido):
            return nome
    for alt in MODELOS_ALTERNATIVOS:
        for nome in instalados:
            if _base(nome) == alt:
                return nome
    return None


# ----------------------------------------------------------------------
# Diagnostico
# ----------------------------------------------------------------------
def diagnosticar(modelo: str = MODELO_PADRAO) -> dict:
    """Retorna {ok, host, modelo_pedido, modelo, instalados, motivo, solucao}."""
    info = {"ok": False, "host": host_ollama(), "modelo_pedido": modelo,
            "modelo": None, "instalados": [], "motivo": "", "solucao": ""}
    try:
        info["instalados"] = listar_modelos()
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        motivo = getattr(e, "reason", e)
        info["motivo"] = f"Servidor Ollama nao responde em {info['host']} ({motivo})."
        info["solucao"] = ("Abra o aplicativo Ollama (ou rode 'ollama serve' num terminal) e "
                           "confira em http://127.0.0.1:11434 se aparece 'Ollama is running'.")
        return info
    except Exception as e:  # resposta inesperada
        info["motivo"] = f"Resposta inesperada do Ollama: {e!r}"
        info["solucao"] = "Atualize o Ollama (https://ollama.com/download)."
        return info
    escolhido = escolher_modelo(modelo, info["instalados"])
    if escolhido is None:
        info["motivo"] = (f"Ollama esta rodando, mas o modelo '{modelo}' nao foi baixado "
                          f"(instalados: {info['instalados'] or 'nenhum'}).")
        info["solucao"] = f"Rode: ollama pull {modelo}"
        return info
    info.update(ok=True, modelo=escolhido,
                motivo="" if _base(escolhido) == _base(modelo)
                else f"'{modelo}' nao encontrado; usando '{escolhido}'.")
    return info


@lru_cache(maxsize=8)
def _diagnostico_cache(modelo: str) -> tuple[bool, str | None, str]:
    d = diagnosticar(modelo)
    return d["ok"], d["modelo"], (d["motivo"] + " " + d["solucao"]).strip()


def llm_disponivel(modelo: str = MODELO_PADRAO) -> bool:
    """True se o servidor responde E ha um modelo utilizavel baixado."""
    return _diagnostico_cache(modelo)[0]


def modelo_em_uso(modelo: str = MODELO_PADRAO) -> str | None:
    return _diagnostico_cache(modelo)[1]


def motivo_indisponivel(modelo: str = MODELO_PADRAO) -> str:
    return _diagnostico_cache(modelo)[2]


def limpar_cache() -> None:
    _diagnostico_cache.cache_clear()


# ----------------------------------------------------------------------
# Chamada
# ----------------------------------------------------------------------
def _extrair_json(texto: str) -> dict:
    texto = re.sub(r"^```(?:json)?|```$", "", texto.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if not match:
        raise ValueError(f"A LLM nao retornou JSON. Resposta bruta: {texto[:300]}")
    return json.loads(match.group(0))


def chamar_llm(prompt: str, modelo: str = MODELO_PADRAO, esperar_json: bool = False,
               temperatura: float = 0.2, tentativas: int = 2) -> str | dict:
    """Envia o prompt ao Ollama (/api/chat) e devolve texto ou dict (esperar_json=True)."""
    ok, escolhido, motivo = _diagnostico_cache(modelo)
    if not ok:
        raise RuntimeError(f"LLM local indisponivel: {motivo}")
    corpo = {"model": escolhido, "stream": False,
             "messages": [{"role": "user", "content": prompt}],
             "options": {"temperature": temperatura, "seed": 42}}
    if esperar_json:
        corpo["format"] = "json"      # forca o Ollama a responder JSON valido
    ultimo_erro: Exception | None = None
    for i in range(max(1, tentativas)):
        try:
            resp = _requisicao("/api/chat", corpo, timeout=TIMEOUT_CHAMADA)
            texto = resp.get("message", {}).get("content", "")
            return _extrair_json(texto) if esperar_json else texto
        except (urllib.error.URLError, OSError, ValueError) as e:
            ultimo_erro = e
            if i + 1 < tentativas:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Falha ao consultar '{escolhido}': {ultimo_erro}")


if __name__ == "__main__":
    d = diagnosticar()
    print(json.dumps(d, indent=2, ensure_ascii=False))
    if d["ok"]:
        print("\nTeste de chamada (a 1a pode demorar enquanto o modelo carrega)...")
        t0 = time.time()
        r = chamar_llm('Responda so com JSON: {"status": "ok"}', esperar_json=True)
        print(f"Resposta: {r}  ({time.time() - t0:.1f}s)")
