"""***************************************************************************************
SCRIPT    : calendario.py
OBJETIVO  : Calendario comercial de semanas, conforme a regra de negocio definida
            pelo grupo (secao 3 do prompt mestre):

              - semana comercial normal: DOMINGO -> SABADO;
              - a PRIMEIRA semana do mes comeca no dia 1 e termina no primeiro
                sabado >= dia 1;
              - a ULTIMA semana termina exatamente no ultimo dia do mes;
              - nenhuma semana ultrapassa o limite do mes;
              - periodo_demanda da semana = data de ENCERRAMENTO da semana.

            Usado agora para o rateio da cota semanal (objetivo_mensal /
            nr_dias de cada semana) e, mais adiante, pelo pipeline semanal.

AUTOR     : Marcos Arambasic
DATA      : AGO/2026
****************************************************************************************/
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

SABADO = 5  # date.weekday(): segunda=0 ... sabado=5, domingo=6


@dataclass(frozen=True)
class SemanaComercial:
    ano: int
    mes: int
    nr_semana: int
    inicio: date
    fim: date

    @property
    def nr_dias(self) -> int:
        return (self.fim - self.inicio).days + 1

    @property
    def periodo_demanda(self) -> date:
        """Por definicao do grupo: a data de encerramento da semana."""
        return self.fim

    @property
    def periodo_mensal(self) -> date:
        """Primeiro dia do mes ao qual a semana pertence."""
        return date(self.ano, self.mes, 1)


def semanas_do_mes(ano: int, mes: int) -> list[SemanaComercial]:
    """
    Decompoe um mes nas suas semanas comerciais.

    A regra garante que nenhuma semana cruza a fronteira do mes: a primeira
    e truncada no inicio e a ultima no fim. Como efeito colateral conhecido e
    aceito pelo grupo, meses que comecam no sabado geram uma primeira semana
    de 1 dia (ex.: 01/04/2023). Isso esta documentado no diagnostico (J7).
    """
    primeiro = date(ano, mes, 1)
    ultimo = date(ano, mes, monthrange(ano, mes)[1])

    semanas: list[SemanaComercial] = []
    inicio = primeiro
    nr = 1
    while inicio <= ultimo:
        # primeiro sabado >= inicio
        fim = inicio + timedelta(days=(SABADO - inicio.weekday()) % 7)
        if fim > ultimo:
            fim = ultimo
        semanas.append(SemanaComercial(ano, mes, nr, inicio, fim))
        inicio = fim + timedelta(days=1)
        nr += 1
    return semanas


def semanas_do_periodo(periodo_inicio: str, periodo_fim: str) -> list[SemanaComercial]:
    """Todas as semanas comerciais entre dois meses (formato 'AAAA-MM'), inclusive."""
    out: list[SemanaComercial] = []
    for p in pd.period_range(periodo_inicio, periodo_fim, freq="M"):
        out.extend(semanas_do_mes(p.year, p.month))
    return out


def meses_do_periodo(periodo_inicio: str, periodo_fim: str) -> list[date]:
    """Primeiro dia de cada mes do periodo."""
    return [date(p.year, p.month, 1) for p in pd.period_range(periodo_inicio, periodo_fim, freq="M")]


def dias_no_mes(ano: int, mes: int) -> int:
    return monthrange(ano, mes)[1]


def calendario_dataframe(periodo_inicio: str, periodo_fim: str) -> pd.DataFrame:
    """
    Calendario comercial completo em DataFrame - serve como dim_calendario
    auditavel e e a base do rateio da cota semanal.
    """
    linhas = []
    for s in semanas_do_periodo(periodo_inicio, periodo_fim):
        linhas.append({
            "nr_ano": s.ano,
            "nr_mes": s.mes,
            "nr_semana": s.nr_semana,
            "data_inicio": s.inicio,
            "data_fim": s.fim,
            "periodo_demanda": s.periodo_demanda,
            "periodo_mensal": s.periodo_mensal,
            "nr_dias": s.nr_dias,
            "dias_no_mes": dias_no_mes(s.ano, s.mes),
            "dias_corridos_ate_fim_semana": (s.fim - s.periodo_mensal).days + 1,
        })
    df = pd.DataFrame(linhas)
    # fracao do mes que cada semana representa -> rateio da cota semanal
    # (regra do Excel: objetivo_mensal / nr_dias de cada semana)
    df["fracao_mes"] = df["nr_dias"] / df["dias_no_mes"]
    return df


def rateio_cota_semanal(cota_mensal: float, ano: int, mes: int) -> list[tuple[int, float]]:
    """
    Distribui a cota mensal entre as semanas proporcionalmente ao numero de
    dias de cada semana (regra informada pelo grupo no Excel).

    Retorna [(nr_semana, cota_da_semana), ...]. A soma e exatamente a cota
    mensal, a menos de erro de ponto flutuante.
    """
    semanas = semanas_do_mes(ano, mes)
    total_dias = dias_no_mes(ano, mes)
    return [(s.nr_semana, cota_mensal * s.nr_dias / total_dias) for s in semanas]


if __name__ == "__main__":
    from .config import PERIODO_INICIO, PERIODO_FIM

    cal = calendario_dataframe(PERIODO_INICIO, PERIODO_FIM)
    print(f"Semanas comerciais em {PERIODO_INICIO}..{PERIODO_FIM}: {len(cal)}")
    print("\nDistribuicao de semanas por mes:")
    print(cal.groupby(["nr_ano", "nr_mes"]).size().value_counts().rename("qtd_meses"))
    print("\nExemplo - abril/2026 (mes que comeca numa quarta):")
    print(cal[(cal.nr_ano == 2026) & (cal.nr_mes == 4)].to_string(index=False))
