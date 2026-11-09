"""***************************************************************************************
SCRIPT    : test_pipeline_sintetico.py
OBJETIVO  : Testes automatizados do pipeline (secao 24 do prompt mestre).
            Cobre calendario comercial, estrutura, integridade, reconciliacao,
            reprodutibilidade e aderencia estatistica aos alvos do Excel.

USO:      pytest tests/ -v
****************************************************************************************/
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.sintetico import config as cfg                       # noqa: E402
from src.sintetico.calendario import (                        # noqa: E402
    calendario_dataframe, dias_no_mes, rateio_cota_semanal, semanas_do_mes,
)
from src.sintetico.calibracao import carregar_calibracao      # noqa: E402
from src.sintetico.cota import calcular_cota_mensal           # noqa: E402
from src.sintetico.gerador import agregar_por_setor, gerar_demanda_mensal  # noqa: E402
from src.sintetico.validacao import validar                   # noqa: E402


# ============================================================================
# Fixtures - a base e gerada uma unica vez e reaproveitada
# ============================================================================
@pytest.fixture(scope="session")
def calib():
    return carregar_calibracao()


@pytest.fixture(scope="session")
def base(calib):
    df, dims, diag = gerar_demanda_mensal(calib=calib)
    cota = calcular_cota_mensal(agregar_por_setor(df, dims))
    return {"demanda": df, "dims": dims, "diag": diag, "cota": cota}


# ============================================================================
# Calendario comercial
# ============================================================================
class TestCalendario:
    def test_semanas_nao_cruzam_fronteira_do_mes(self):
        for ano in [2023, 2024, 2025, 2026]:
            for mes in range(1, 13):
                for s in semanas_do_mes(ano, mes):
                    assert s.inicio.month == mes and s.inicio.year == ano
                    assert s.fim.month == mes and s.fim.year == ano

    def test_primeira_semana_comeca_no_dia_1(self):
        for ano in [2023, 2026]:
            for mes in range(1, 13):
                assert semanas_do_mes(ano, mes)[0].inicio == date(ano, mes, 1)

    def test_ultima_semana_termina_no_ultimo_dia(self):
        for ano in [2023, 2026]:
            for mes in range(1, 13):
                semanas = semanas_do_mes(ano, mes)
                assert semanas[-1].fim == date(ano, mes, dias_no_mes(ano, mes))

    def test_semanas_sao_contiguas_e_cobrem_o_mes(self):
        for ano in [2024, 2025]:
            for mes in range(1, 13):
                semanas = semanas_do_mes(ano, mes)
                assert sum(s.nr_dias for s in semanas) == dias_no_mes(ano, mes)
                for a, b in zip(semanas, semanas[1:]):
                    assert (b.inicio - a.fim).days == 1

    def test_semanas_intermediarias_terminam_no_sabado(self):
        for s in semanas_do_mes(2026, 4)[1:-1]:
            assert s.fim.weekday() == 5, "semana intermediaria deve fechar no sabado"

    def test_periodo_demanda_e_o_fim_da_semana(self):
        for s in semanas_do_mes(2026, 4):
            assert s.periodo_demanda == s.fim

    def test_exemplo_documentado_abril_2026(self):
        """Mes que comeca numa quarta - exemplo do prompt mestre."""
        s = semanas_do_mes(2026, 4)
        assert (s[0].inicio, s[0].fim) == (date(2026, 4, 1), date(2026, 4, 4))
        assert (s[1].inicio, s[1].fim) == (date(2026, 4, 5), date(2026, 4, 11))
        assert s[-1].fim == date(2026, 4, 30)

    def test_total_de_semanas_no_periodo(self):
        cal = calendario_dataframe(cfg.PERIODO_INICIO, cfg.PERIODO_FIM)
        assert len(cal) == 229
        assert cal.groupby(["nr_ano", "nr_mes"]).ngroups == 44

    def test_rateio_da_cota_semanal_soma_a_cota_mensal(self):
        for ano, mes in [(2023, 1), (2024, 2), (2026, 4), (2026, 8)]:
            rateio = rateio_cota_semanal(1000.0, ano, mes)
            assert sum(v for _, v in rateio) == pytest.approx(1000.0)


# ============================================================================
# Estrutura (secao 24: contagens corretas)
# ============================================================================
class TestEstrutura:
    def test_numero_de_meses(self, base):
        assert base["demanda"].periodo_demanda.nunique() == 44

    def test_numero_de_produtos(self, base):
        assert base["demanda"].id_produto.nunique() == cfg.N_PRODUTOS == 30

    def test_divisao_proprios_concorrentes(self, base):
        p = base["dims"].produtos
        assert (p.tipo_produto == "PROPRIO").sum() == 10
        assert (p.tipo_produto == "CONCORRENTE").sum() == 20

    def test_grupos_dos_produtos_proprios(self, base):
        p = base["dims"].produtos
        grupos = p[p.tipo_produto == "PROPRIO"].grupo_produto.value_counts()
        assert len(grupos) == 2 and set(grupos.values) == {5}

    def test_numero_de_outlets_e_setores(self, base):
        assert len(base["dims"].outlets) == 2700
        assert len(base["dims"].setores) == 27
        assert len(base["dims"].ufs) == 27

    def test_cem_outlets_por_uf(self, base):
        assert (base["dims"].outlets.groupby("id_uf").size() == 100).all()

    def test_densidade_proxima_do_alvo(self, base):
        assert abs(base["diag"]["densidade"] - cfg.DENSIDADE_ALVO) < 0.05


# ============================================================================
# Integridade
# ============================================================================
class TestIntegridade:
    def test_demanda_nao_negativa(self, base):
        assert (base["demanda"].demanda_unidades >= 0).all()

    def test_cota_nao_negativa(self, base):
        assert (base["cota"].cota_unidades >= 0).all()

    def test_sem_nulos(self, base):
        assert base["demanda"].isna().sum().sum() == 0

    def test_sem_duplicidade_no_grao(self, base):
        d = base["demanda"]
        assert not d.duplicated(subset=["periodo_demanda", "id_produto", "id_outlet"]).any()

    def test_cota_sem_duplicidade_no_grao(self, base):
        c = base["cota"]
        assert not c.duplicated(subset=["periodo_demanda", "id_setor", "id_produto"]).any()

    def test_fks_validas(self, base):
        d, dims = base["demanda"], base["dims"]
        assert d.id_produto.isin(dims.produtos.id_produto).all()
        assert d.id_outlet.isin(dims.outlets.id_outlet).all()
        assert d.id_uf.isin(dims.ufs.id_uf).all()

    def test_uf_coerente_com_outlet(self, base):
        d, dims = base["demanda"], base["dims"]
        mapa = dims.outlets.set_index("id_outlet")["id_uf"]
        assert (d.id_outlet.map(mapa) == d.id_uf).all()

    def test_market_share_entre_0_e_1(self, base):
        d, dims = base["demanda"], base["dims"]
        tipo = dims.produtos.set_index("id_produto")["tipo_produto"]
        g = (d.assign(t=d.id_produto.map(tipo))
             .groupby(["periodo_demanda", "id_uf", "t"])["demanda_unidades"]
             .sum().unstack(fill_value=0))
        ms = g["PROPRIO"] / g.sum(axis=1)
        assert ((ms >= 0) & (ms <= 1)).all()

    def test_cota_apenas_de_produtos_proprios(self, base):
        proprios = base["dims"].produtos.query("tipo_produto == 'PROPRIO'").id_produto
        assert base["cota"].id_produto.isin(proprios).all()


# ============================================================================
# Reconciliacao (secao 22)
# ============================================================================
class TestReconciliacao:
    def test_soma_outlets_igual_setor(self, base):
        d, dims = base["demanda"], base["dims"]
        por_setor = agregar_por_setor(d, dims, apenas_proprios=False)
        assert por_setor.demanda_unidades.sum() == pytest.approx(
            d.demanda_unidades.sum(), rel=cfg.TOLERANCIA_RECONCILIACAO
        )

    def test_soma_produtos_igual_mercado(self, base):
        d = base["demanda"]
        por_produto = d.groupby(["periodo_demanda", "id_produto"])["demanda_unidades"].sum()
        assert por_produto.sum() == pytest.approx(
            d.demanda_unidades.sum(), rel=cfg.TOLERANCIA_RECONCILIACAO
        )

    def test_todo_outlet_pertence_a_um_setor(self, base):
        dims = base["dims"]
        assert dims.alocacao.id_outlet.nunique() == len(dims.outlets)
        assert dims.alocacao.id_setor.notna().all()


# ============================================================================
# Reprodutibilidade (secao 21)
# ============================================================================
class TestReprodutibilidade:
    def test_mesma_seed_gera_base_identica(self, calib):
        a, _, _ = gerar_demanda_mensal(seed=123, calib=calib)
        b, _, _ = gerar_demanda_mensal(seed=123, calib=calib)
        pd.testing.assert_frame_equal(a, b)

    def test_seeds_diferentes_geram_bases_diferentes(self, calib):
        a, _, _ = gerar_demanda_mensal(seed=1, calib=calib)
        b, _, _ = gerar_demanda_mensal(seed=2, calib=calib)
        assert not a.demanda_unidades.equals(b.demanda_unidades)


# ============================================================================
# Aderencia estatistica aos alvos do Excel (secao 22)
# ============================================================================
class TestAderenciaEstatistica:
    @pytest.fixture(scope="class")
    def janela(self, base, calib):
        d = base["demanda"].copy()
        d["periodo_demanda"] = pd.to_datetime(d.periodo_demanda)
        tipo = base["dims"].produtos.set_index("id_produto")["tipo_produto"]
        d["tipo"] = d.id_produto.map(tipo)
        j = calib["janela_observada"]
        return d[(d.periodo_demanda >= pd.Timestamp(j["inicio"]))
                 & (d.periodo_demanda <= pd.Timestamp(j["fim"]))]

    def test_cagr_mercado_na_convencao_do_excel(self, janela, calib):
        nac = janela.groupby(["periodo_demanda", "tipo"])["demanda_unidades"].sum().unstack()
        nac["MERCADO"] = nac.sum(axis=1)
        jan = calib["janela_observada"]["meses_por_ano"]
        anos = sorted(int(a) for a in jan)

        def media(ano):
            m = jan[str(ano)]
            return nac[(nac.index.year == ano) & (nac.index.month.isin(m))]["MERCADO"].mean()

        cagr = (media(anos[-1]) / media(anos[0])) ** (1 / (anos[-1] - anos[0])) - 1
        assert abs(cagr - calib["tendencia"]["cagr_excel_mercado"]) < cfg.TOLERANCIA_TENDENCIA

    def test_cv_temporal_do_mercado(self, janela, calib):
        g = janela.groupby(["id_uf", "periodo_demanda"])["demanda_unidades"].sum()
        cv = g.groupby(level=0).apply(lambda s: s.std(ddof=1) / s.mean()).mean()
        alvo = calib["variabilidade"]["cv_bruto_mercado"]
        assert abs(cv - alvo) / alvo < cfg.TOLERANCIA_DISTRIBUICAO

    def test_market_share_medio(self, janela):
        g = (janela.groupby(["id_uf", "periodo_demanda", "tipo"])["demanda_unidades"]
             .sum().unstack(fill_value=0))
        ms = g["PROPRIO"] / g.sum(axis=1)
        alvo = cfg.ALVOS_DISTRIBUICAO["market_share"]["media"]
        assert abs(ms.mean() - alvo) / alvo < cfg.TOLERANCIA_DISTRIBUICAO

    def test_erosao_de_share_e_monotona(self, base):
        d = base["demanda"].copy()
        d["periodo_demanda"] = pd.to_datetime(d.periodo_demanda)
        tipo = base["dims"].produtos.set_index("id_produto")["tipo_produto"]
        d["tipo"] = d.id_produto.map(tipo)
        por_ano = d.groupby([d.periodo_demanda.dt.year, "tipo"])["demanda_unidades"].sum().unstack()
        share = por_ano["PROPRIO"] / por_ano.sum(axis=1)
        assert (share.sort_index().diff().dropna() < 0).all(), \
            "o share dos proprios deve cair ano a ano"

    def test_concentracao_pareto_intra_uf(self, base):
        o = base["demanda"].groupby(["id_uf", "id_outlet"])["demanda_unidades"].sum()
        topo = [g.sort_values(ascending=False).head(20).sum() / g.sum()
                for _, g in o.groupby(level=0)]
        assert abs(np.mean(topo) - cfg.PARETO_SHARE_ALVO) < 0.10

    def test_sazonalidade_media_anual_proxima_de_um(self, calib):
        for chave in ["proprios", "mercado"]:
            for uf, perfil in calib["sazonalidade"][chave].items():
                media = np.mean(list(perfil.values()))
                assert abs(media - 1.0) < 1e-6, f"{chave}/{uf}: media {media}"


# ============================================================================
# Cota (secoes 11 e 13)
# ============================================================================
class TestCota:
    def test_razao_cota_demanda_bate_com_o_excel(self, base):
        c = base["cota"].copy()
        c["periodo_demanda"] = pd.to_datetime(c.periodo_demanda)
        d = agregar_por_setor(base["demanda"], base["dims"])
        d["periodo_demanda"] = pd.to_datetime(d.periodo_demanda)
        m = c.merge(d, on=["periodo_demanda", "id_setor", "id_produto"])
        razao = (m.cota_unidades / m.demanda_unidades.replace(0, np.nan)).dropna()
        assert abs(razao.median() - (1 + cfg.COTA_CRESCIMENTO)) < 0.10

    def test_bootstrap_marcado_apenas_nos_primeiros_meses(self, base):
        c = base["cota"]
        boot = pd.to_datetime(c[c.bootstrap].periodo_demanda)
        if len(boot):
            assert boot.max() < pd.Timestamp("2023-08-01")

    def test_cota_respeita_o_grao(self, base):
        c = base["cota"]
        assert c.id_setor.nunique() == 27
        assert c.id_produto.nunique() == 10


# ============================================================================
# Fidelity score
# ============================================================================
class TestKpiMensal:
    @pytest.fixture(scope="class")
    def kpi(self, base):
        from src.sintetico.kpi import calcular_analise_kpi
        return calcular_analise_kpi(base["demanda"], base["dims"])

    def test_grao_e_volume(self, kpi, base):
        assert len(kpi) == len(base["demanda"])
        assert not kpi.duplicated(subset=["periodo_demanda", "id_produto", "id_outlet"]).any()

    def test_market_share_entre_0_e_1(self, kpi):
        assert ((kpi.mrkt_share >= 0) & (kpi.mrkt_share <= 1)).all()

    def test_sem_infinitos_nem_nulos(self, kpi):
        for c in ["mrkt_share", "cagr_medio", "cv", "indice_sazonal"]:
            assert np.isfinite(kpi[c]).all(), f"{c} tem inf/NaN"

    def test_market_share_bate_com_a_demanda(self, kpi):
        ativo = kpi[(kpi.demanda_mercado > 0) & (kpi.demanda_produto > 0)].head(5000)
        esperado = ativo.demanda_produto / ativo.demanda_mercado
        assert np.allclose(ativo.mrkt_share, esperado)

    def test_cagr_zerado_apenas_no_primeiro_ano(self, kpi):
        """Sem 2 anos de historico nao ha CAGR - so 2023 pode vir zerado."""
        anos_com_zero = kpi.loc[kpi.cagr_medio == 0, "periodo_demanda"].dt.year.unique()
        assert set(anos_com_zero) <= {2023}

    def test_cv_nao_negativo(self, kpi):
        assert (kpi.cv >= 0).all()

    def test_faixas_sugeridas_sem_degeneracao(self, kpi):
        from src.sintetico.kpi import sugerir_faixas_kpi
        faixas = sugerir_faixas_kpi(
            kpi[kpi.demanda_produto > 0],
            kpis=["mrkt_share", "cagr_medio", "cv", "indice_sazonal"],
        )
        assert (faixas.valor_inicial < faixas.valor_final).all(), \
            "faixa com inicio == fim nunca captura nenhuma linha"
        assert faixas.groupby("desc_kpi").size().eq(4).all()


class TestPipelineSemanal:
    @pytest.fixture(scope="class")
    def semanal(self, base):
        from src.sintetico.pipeline_semanal import desagregar_semanal
        return desagregar_semanal(base["demanda"])

    def test_reconciliacao_semanal_igual_mensal(self, semanal, base):
        soma = semanal.groupby(
            ["periodo_mensal", "id_produto", "id_outlet"]
        )["demanda_unidades"].sum().sum()
        assert soma == pytest.approx(base["demanda"].demanda_unidades.sum(), rel=1e-9)

    def test_total_de_semanas(self, semanal):
        assert semanal.periodo_demanda.nunique() == 229

    def test_demanda_semanal_inteira_e_nao_negativa(self, semanal):
        assert (semanal.demanda_unidades >= 0).all()
        assert np.allclose(semanal.demanda_unidades % 1, 0)

    def test_semana_nunca_cruza_o_mes(self, semanal):
        assert (semanal.periodo_demanda.dt.to_period("M")
                == semanal.periodo_mensal.dt.to_period("M")).all()

    @pytest.fixture(scope="class")
    def alocado(self, semanal, base):
        from src.sintetico.pipeline_semanal import alocar_semanal
        proprios = base["dims"].produtos.query("tipo_produto == 'PROPRIO'").id_produto
        return alocar_semanal(semanal[semanal.id_produto.isin(proprios)], base["dims"])

    def test_acumulado_reinicia_a_cada_mes(self, alocado):
        g = alocado.sort_values(["periodo_mensal", "nr_semana"]).groupby(
            ["id_produto", "id_alocacao", "periodo_mensal"], sort=False
        )
        for _, x in list(g)[:400]:
            assert x.demanda_unidades_acumulada.iloc[0] == x.demanda_unidades.iloc[0]

    def test_acumulado_e_monotonico_dentro_do_mes(self, alocado):
        g = alocado.sort_values(["periodo_mensal", "nr_semana"]).groupby(
            ["id_produto", "id_alocacao", "periodo_mensal"], sort=False
        )
        for _, x in list(g)[:400]:
            assert x.demanda_unidades_acumulada.is_monotonic_increasing

    def test_acumulado_final_igual_soma_das_semanas(self, alocado):
        g = alocado.sort_values(["periodo_mensal", "nr_semana"]).groupby(
            ["id_produto", "id_alocacao", "periodo_mensal"], sort=False
        )
        for _, x in list(g)[:400]:
            assert x.demanda_unidades_acumulada.iloc[-1] == pytest.approx(
                x.demanda_unidades.sum()
            )

    def test_alocado_cobre_apenas_proprios(self, alocado, base):
        proprios = base["dims"].produtos.query("tipo_produto == 'PROPRIO'").id_produto
        assert alocado.id_produto.isin(proprios).all()


class TestClassificacao:
    def test_classificar_respeita_as_faixas(self):
        from src.sintetico.pipeline_semanal import classificar

        faixas = pd.DataFrame([
            {"desc_kpi": k, "valor_inicial": ini, "valor_final": fim,
             "desc_grupo": g, "cd_grupo_padrao": cd}
            for k in ["cobertura_cota_projetada", "mrkt_share", "cagr_medio",
                      "cv", "indice_sazonal"]
            for ini, fim, g, cd in [
                (-999999.0, 0.25, "baixo", 1), (0.25, 0.50, "medio", 2),
                (0.50, 0.75, "alto", 3), (0.75, 999999.0, "muito_alto", 4),
            ]
        ])
        analise = pd.DataFrame({
            "periodo_demanda": pd.to_datetime(["2025-03-08"] * 4),
            "nr_semana": [1, 1, 1, 1],
            "id_produto": [1, 1, 1, 1], "id_mercado": [1, 1, 1, 1],
            "id_setor": [1, 1, 1, 1], "id_alocacao": [1, 2, 3, 4], "id_uf": [1, 1, 1, 1],
            "cobertura_cota_projetada": [0.10, 0.40, 0.60, 0.90],
            "mrkt_share": [0.10, 0.40, 0.60, 0.90],
            "cagr_medio": [0.10, 0.40, 0.60, 0.90],
            "cv": [0.10, 0.40, 0.60, 0.90],
            "indice_sazonal": [0.10, 0.40, 0.60, 0.90],
        })
        out = classificar(analise, faixas)
        assert out.cd_mrkt_share.tolist() == [1, 2, 3, 4]
        assert out.cd_cobertura_cota_projetada.tolist() == [1, 2, 3, 4]
        assert out.nr_ano.unique().tolist() == [2025]

    def test_erro_claro_se_faltar_kpi_nas_faixas(self):
        from src.sintetico.pipeline_semanal import classificar
        faixas = pd.DataFrame([{
            "desc_kpi": "mrkt_share", "valor_inicial": 0.0, "valor_final": 1.0,
            "desc_grupo": "unico", "cd_grupo_padrao": 1,
        }])
        analise = pd.DataFrame({
            "periodo_demanda": pd.to_datetime(["2025-03-08"]), "nr_semana": [1],
            "id_produto": [1], "id_mercado": [1], "id_setor": [1],
            "id_alocacao": [1], "id_uf": [1],
            "cobertura_cota_projetada": [0.5], "mrkt_share": [0.5],
            "cagr_medio": [0.5], "cv": [0.5], "indice_sazonal": [0.5],
        })
        with pytest.raises(ValueError):
            classificar(analise, faixas)


class TestContratoDeExportacao:
    """Garante que cada tabela sai com as colunas exatas do DDL."""

    def test_colunas_das_tabelas_mensais(self, base):
        from src.sintetico.exportacao import COLUNAS_POR_TABELA, preparar_para_csv
        d = preparar_para_csv(base["demanda"], "fato_demanda_aberta_mensal")
        assert list(d.columns) == COLUNAS_POR_TABELA["fato_demanda_aberta_mensal"]
        c = preparar_para_csv(base["cota"], "fato_cota_mensal")
        assert list(c.columns) == COLUNAS_POR_TABELA["fato_cota_mensal"]

    def test_data_sai_sem_hora(self, base):
        from src.sintetico.exportacao import preparar_para_csv
        d = preparar_para_csv(base["demanda"], "fato_demanda_aberta_mensal")
        assert d.periodo_demanda.str.match(r"^\d{4}-\d{2}-\d{2}$").all()

    def test_demanda_sai_como_inteiro(self, base):
        from src.sintetico.exportacao import preparar_para_csv
        d = preparar_para_csv(base["demanda"], "fato_demanda_aberta_mensal")
        assert d.demanda_unidades.dtype.kind == "i"

    def test_tabela_desconhecida_da_erro(self, base):
        from src.sintetico.exportacao import preparar_para_csv
        with pytest.raises(ValueError):
            preparar_para_csv(base["demanda"], "tabela_que_nao_existe")


class TestKpiFinal:
    """fato_analise_kpi_final - base de treino do Agente 3."""

    def test_classifica_e_traz_setor(self):
        from src.sintetico.kpi_final import COLUNAS_KPI_FINAL, montar_kpi_final
        kpi = pd.DataFrame({
            "periodo_demanda": ["2025-01-01"] * 3, "id_produto": [1, 1, 2],
            "grupo": ["A"] * 3, "id_mercado": [1] * 3, "id_outlet": [10, 11, 10],
            "id_uf": [1] * 3, "demanda_produto": [5, 1, 3], "demanda_mercado": [50, 40, 50],
            "mrkt_share": [0.10, 0.025, 0.06], "cagr_medio": [-0.1, 0.0, 0.6],
            "cv": [0.05, 0.3, 0.1], "indice_sazonal": [0.7, 1.0, 1.3],
        })
        aloc = pd.DataFrame({"id_alocacao": [100, 101], "id_outlet": [10, 11],
                             "id_uf": [1, 1], "id_setor": [7, 7]})
        faixas = pd.read_csv(RAIZ / "data" / "sintetico" / "dim_classifica_kpis.csv") \
            if (RAIZ / "data" / "sintetico" / "dim_classifica_kpis.csv").exists() else None
        if faixas is None:
            pytest.skip("dim_classifica_kpis.csv nao gerada")
        out = montar_kpi_final(kpi, aloc, faixas)
        assert list(out.columns) == COLUNAS_KPI_FINAL
        assert (out["id_setor"] == 7).all()
        assert out.loc[0, "cd_mrkt_share"] == 4 and out.loc[1, "cd_mrkt_share"] == 1
        assert out.loc[0, "cd_cagr_medio"] == 1 and out.loc[2, "cd_cagr_medio"] == 4

    def test_outlet_em_duas_alocacoes_falha(self):
        from src.sintetico.kpi_final import montar_kpi_final
        aloc = pd.DataFrame({"id_alocacao": [1, 2], "id_outlet": [10, 10],
                             "id_uf": [1, 1], "id_setor": [1, 1]})
        with pytest.raises(ValueError):
            montar_kpi_final(pd.DataFrame(), aloc, pd.DataFrame(
                columns=["desc_kpi", "valor_inicial", "valor_final", "cd_grupo_padrao"]))


class TestAgente3:
    """Agente 3 - Oportunidades de Mercado. Painel sintetico pequeno."""

    @pytest.fixture(scope="class")
    def kpi_final(self):
        rng = np.random.default_rng(7)
        linhas = []
        meses = pd.date_range("2024-01-01", periods=24, freq="MS")
        for t, m in enumerate(meses):
            for uf in (1, 2):
                for k in range(6):
                    aloc = uf * 100 + k
                    mercado = 1000 * (1 + 0.02 * t) * (1 + 0.1 * np.sin(m.month)) * (1 + 0.2 * k)
                    shares = {1: 0.10 - 0.001 * t + 0.01 * k, 2: 0.05 + 0.0005 * t,
                              3: 0.03, 11: 0.30 + 0.002 * t, 12: 0.20}
                    for prod, s in shares.items():
                        s = max(0.0, s + rng.normal(0, 0.004))
                        linhas.append({
                            "periodo_demanda": m, "id_produto": prod,
                            "grupo": "Grupo A" if prod < 10 else "Grupo C",
                            "id_mercado": 1, "id_setor": uf, "id_alocacao": aloc,
                            "id_outlet": aloc, "id_uf": uf,
                            "demanda_produto": s * mercado, "demanda_mercado": mercado,
                            "mrkt_share": s, "cagr_medio": -0.05 if prod < 10 else 0.3,
                            "cv": 0.1, "indice_sazonal": 1 + 0.2 * np.sin(m.month),
                            "cd_mrkt_share": 1 + int(s > 0.04) + int(s > 0.06) + int(s > 0.09),
                            "cd_cagr_medio": 1 if prod < 10 else 3,
                            "cd_cv": 2, "cd_indice_sazonal": 1 + int(np.sin(m.month) > -0.5)
                            + int(np.sin(m.month) > 0) + int(np.sin(m.month) > 0.7),
                        })
        return pd.DataFrame(linhas)

    @pytest.fixture(scope="class")
    def produtos(self):
        return pd.DataFrame({"id_produto": [1, 2, 3, 11, 12],
                             "tipo_produto": ["PROPRIO"] * 3 + ["CONCORRENTE"] * 2})

    @pytest.fixture(scope="class")
    def painel(self, kpi_final, produtos):
        from src.agentes.agente3_oportunidades_mercado import montar_dataset
        return montar_dataset(kpi_final, produtos)

    @pytest.fixture(scope="class")
    def agente(self, painel):
        from src.agentes.agente3_oportunidades_mercado import AgenteOportunidadesMercado
        ag = AgenteOportunidadesMercado(hiperparametros={"n_estimators": 50})
        ag.treinar(painel, meses_teste=3)
        return ag

    @pytest.fixture(scope="class")
    def recs(self, agente):
        return agente.recomendar()

    @pytest.fixture(scope="class")
    def saida(self, agente, recs):
        return agente.exportar_json(recs, top_por_setor=5)

    def test_painel_so_tem_proprios_e_e_completo(self, painel):
        d = painel.dados
        assert set(d.id_produto) == {1, 2, 3}
        assert len(d) == 3 * 12 * 24

    def test_alvo_e_a_variacao_do_mes_seguinte(self, painel):
        from src.agentes.agente3_oportunidades_mercado import ALVO
        d = painel.dados.sort_values(["id_produto", "id_alocacao", "periodo_demanda"])
        g = d[(d.id_produto == 1) & (d.id_alocacao == 100)]
        esperado = g["mrkt_share"].shift(-1) - g["mrkt_share"]
        assert np.allclose(g[ALVO].iloc[:-1], esperado.iloc[:-1])
        assert np.isnan(g[ALVO].iloc[-1])

    def test_sem_vazamento_do_indice_sazonal(self, painel, kpi_final):
        """A variavel sazonal usa t-11, nunca o mes corrente."""
        from src.agentes.agente3_oportunidades_mercado import FEATURES
        assert "indice_sazonal" not in FEATURES
        d = painel.dados[(painel.dados.id_produto == 1) & (painel.dados.id_alocacao == 100)]
        assert d["indice_sazonal_previsto"].iloc[:11].isna().all()

    def test_split_temporal(self, painel):
        from src.agentes.agente3_oportunidades_mercado import dividir_treino_teste
        treino, teste, corte = dividir_treino_teste(painel, 3)
        assert treino.periodo_demanda.max() < corte <= teste.periodo_demanda.min()

    def test_metricas_e_baselines_finitas(self, agente):
        r = agente.resultado
        for m in ("mae", "rmse", "r2"):
            assert np.isfinite(r.metricas[m])
            assert np.isfinite(r.baselines["persistencia"][m])

    def test_importancia_cobre_todas_as_variaveis(self, agente):
        from src.agentes.agente3_oportunidades_mercado import FEATURES
        assert {v["variavel"] for v in agente.resultado.importancia} == set(FEATURES)

    def test_recomendacoes_coerentes(self, recs):
        from src.agentes.agente3_oportunidades_mercado import TIPOS
        assert set(recs.tipo_recomendacao) <= set(TIPOS)
        assert (recs.oportunidade_un >= 0).all() and (recs.risco_un >= 0).all()
        assert recs.share_previsto.between(0, 1).all()
        assert (recs.loc[recs.natureza == "neutro", "valor_em_jogo_un"] == 0).all()

    def test_recomendar_sem_treino_da_erro(self):
        from src.agentes.agente3_oportunidades_mercado import AgenteOportunidadesMercado
        with pytest.raises(RuntimeError):
            AgenteOportunidadesMercado().recomendar()

    def test_contrato_json_valido(self, saida):
        from src.agentes.agente3_oportunidades_mercado import validar_saida
        assert validar_saida(saida) == []

    def test_json_serializavel_e_chaves_para_agente4(self, saida):
        import json as _json
        d = _json.loads(_json.dumps(saida, ensure_ascii=False))
        assert d["instrucoes_agente4"]["filtro_noticias"]["nivel_impacto"] == ["alto", "medio"]
        for s in d["setores"]:
            for it in s["itens"]:
                assert {"id_produto", "id_mercado", "ids_concorrentes"} <= set(it["chaves_noticia"])
                assert set(it["chaves_noticia"]["ids_concorrentes"]) <= {11, 12}

    def test_validador_detecta_erros(self, saida):
        import copy
        from src.agentes.agente3_oportunidades_mercado import validar_saida
        assert any("resumo" in e for e in validar_saida(
            {k: v for k, v in saida.items() if k != "resumo"}))
        q = copy.deepcopy(saida)
        q["schema_version"] = "0.1"
        assert any("schema_version" in e for e in validar_saida(q))
        if q["setores"] and q["setores"][0]["itens"]:
            q = copy.deepcopy(saida)
            q["setores"][0]["itens"][0]["natureza"] = "risco" if \
                q["setores"][0]["itens"][0]["natureza"] != "risco" else "oportunidade"
            assert any("natureza" in e for e in validar_saida(q))


class TestFidelityScore:
    def test_score_acima_do_minimo(self, base, calib):
        rel = validar(base["demanda"], base["cota"], base["dims"], calib)
        score = rel.fidelity_score()
        assert score >= 80.0, f"fidelity score {score} abaixo do minimo aceitavel"

    def test_sem_falhas_criticas(self, base, calib):
        rel = validar(base["demanda"], base["cota"], base["dims"], calib)
        criticas = rel.df().query("dimensao in ['integridade','reconciliacao'] and score < 100")
        assert criticas.empty, f"falhas criticas:\n{criticas[['metrica', 'observacao']]}"


class TestNoticiasEAgentes4e5:
    """Classificacao de noticias (Agentes 1/2) e cruzamento (Agentes 4 e 5)."""

    @pytest.fixture(scope="class")
    def saida_a3(self):
        item = lambda i, tipo, nat, un, grupo="Grupo A": {
            "id_item": f"2026-10|S1|A{i}|P1", "id_setor": 1, "id_alocacao": i, "id_outlet": i,
            "id_uf": 1, "id_produto": 1, "id_mercado": 1, "grupo": grupo,
            "tipo_recomendacao": tipo, "natureza": nat, "prioridade": "alta",
            "acao_sugerida": "x", "previsao": {"valor_em_jogo_un": un},
            "concorrentes_relevantes": [{"id_produto": 11, "mrkt_share": 0.2}],
        }
        from src.agentes.agente3_oportunidades_mercado import REGRAS_SENSIBILIDADE
        return {
            "periodo_previsto": "2026-10",
            "instrucoes_agente4": {"filtro_noticias": {"nivel_impacto": ["alto", "medio"]},
                                   "regras_de_sensibilidade": REGRAS_SENSIBILIDADE},
            "setores": [{"id_setor": 1, "id_uf": 1, "itens": [
                item(1, "DEFENDER_SHARE", "risco", 100),
                item(2, "AMPLIAR_PENETRACAO", "oportunidade", 50),
                item(3, "CAPTURAR_CRESCIMENTO", "oportunidade", 80, grupo="Grupo B"),
            ]}],
        }

    @pytest.fixture(scope="class")
    def noticias(self):
        return pd.DataFrame([
            {"id_noticia": "n1", "titulo": "Concorrente lanca nova versao", "descricao": "",
             "url": "u1", "fonte": "f", "data_publicacao": "2026-09-20", "nivel_impacto": "alto",
             "sentimento": "positivo", "score_sentimento": 0.5, "escopo": "concorrente",
             "grupos_alvo": "", "categoria": "concorrencia", "origem_classificacao": "regras"},
            {"id_noticia": "n2", "titulo": "Empagliflozina tem nova indicacao", "descricao": "",
             "url": "u2", "fonte": "f", "data_publicacao": "2026-09-01", "nivel_impacto": "medio",
             "sentimento": "positivo", "score_sentimento": 0.5, "escopo": "proprio",
             "grupos_alvo": "Grupo A", "categoria": "regulatorio", "origem_classificacao": "regras"},
            {"id_noticia": "n3", "titulo": "irrelevante", "descricao": "", "url": "u3", "fonte": "f",
             "data_publicacao": "2026-09-01", "nivel_impacto": "baixo", "sentimento": "neutro",
             "score_sentimento": 0, "escopo": "mercado", "grupos_alvo": "", "categoria": "geral",
             "origem_classificacao": "regras"},
        ])

    def test_vinculo_de_entidades(self):
        from src.noticias.classificacao import MAPA_PADRAO, normalizar, vincular_entidades
        esc, grupos, _ = vincular_entidades(normalizar("Jardiance (empagliflozina) é aprovado"), MAPA_PADRAO)
        assert esc == "proprio" and grupos == ["Grupo A"]
        assert vincular_entidades(normalizar("Ozempic em falta"), MAPA_PADRAO)[0] == "concorrente"
        assert vincular_entidades(normalizar("Anvisa publica norma"), MAPA_PADRAO)[0] == "mercado"
        # termo curto casa so como palavra inteira: "sus" nao casa com "suspensao"
        assert vincular_entidades(normalizar("suspensao de jogo de futebol"), MAPA_PADRAO)[0] == "nenhum"

    def test_sentimento(self):
        from src.noticias.classificacao import sentimento_heuristico
        assert sentimento_heuristico("anvisa aprova novo medicamento com beneficio")[0] == "positivo"
        assert sentimento_heuristico("recall por risco de efeito adverso")[0] == "negativo"

    def test_agente4_aplica_regras_e_filtra_baixo(self, saida_a3, noticias):
        from src.agentes.agente4_riscos_oportunidades import AgenteRiscosOportunidades, validar_saida
        s = AgenteRiscosOportunidades(saida_a3).cruzar(noticias)
        assert validar_saida(s) == []
        achados = s["setores"][0]["achados"]
        assert {a["id_noticia"] for a in achados} == {"n1", "n2"}          # n3 e baixo
        # noticia positiva de concorrente CONFIRMA o risco de DEFENDER_SHARE
        a = next(x for x in achados if x["id_noticia"] == "n1" and x["tipo_recomendacao"] == "DEFENDER_SHARE")
        assert a["multiplicador"] > 1 and a["categoria"] == "risco_imediato"
        # noticia de produto proprio do Grupo A nao liga com item do Grupo B
        assert not any(x["id_noticia"] == "n2" and x["id_alocacao"] == 3 for x in achados)

    def test_agente5_gera_destaques_validos(self, saida_a3, noticias):
        from src.agentes.agente4_riscos_oportunidades import AgenteRiscosOportunidades
        from src.agentes.agente5_recomendacoes import AgenteRecomendacoes, validar_saida
        s4 = AgenteRiscosOportunidades(saida_a3).cruzar(noticias)
        s5 = AgenteRecomendacoes().gerar(s4, n_noticias=4)
        assert validar_saida(s5) == []
        d = s5["setores"][0]["destaques"]
        assert 1 <= len(d) <= 4
        assert [x["posicao"] for x in d] == list(range(1, len(d) + 1))
        assert all(d[i]["score"] >= d[i + 1]["score"] for i in range(len(d) - 1))


class TestExtensaoSemanal:
    """Criacao de meses alem da base (executar_semana sem trava de horizonte)."""

    @pytest.fixture(scope="class")
    def gerado(self):
        from src.sintetico.gerador import gerar_com_estado
        return gerar_com_estado(periodo_inicio="2023-01", periodo_fim="2024-06")

    def test_estado_nao_altera_a_base(self, gerado):
        from src.sintetico.gerador import gerar_demanda_mensal
        df, *_ = gerado
        df2, *_ = gerar_demanda_mensal(periodo_inicio="2023-01", periodo_fim="2024-06")
        assert (df["demanda_unidades"].to_numpy() == df2["demanda_unidades"].to_numpy()).all()

    def test_mes_criado_e_deterministico(self, gerado):
        from datetime import date
        from src.sintetico.extensao import gerar_meses
        _, _, _, est = gerado
        a = gerar_meses(est, date(2024, 9, 1))
        b = gerar_meses(est, date(2024, 8, 1))
        ma = a[a.periodo_demanda == "2024-08-01"]["demanda_unidades"].to_numpy()
        mb = b[b.periodo_demanda == "2024-08-01"]["demanda_unidades"].to_numpy()
        assert (ma == mb).all() and len(ma) > 0
        assert set(a.periodo_demanda.dt.strftime("%Y-%m")) == {"2024-07", "2024-08", "2024-09"}

    def test_mes_criado_mantem_pares_e_nivel(self, gerado):
        from datetime import date
        from src.sintetico.extensao import gerar_meses
        df, _, _, est = gerado
        novo = gerar_meses(est, date(2024, 7, 1))
        pares_base = set(map(tuple, df[["id_produto", "id_outlet"]].drop_duplicates().to_numpy()))
        assert set(map(tuple, novo[["id_produto", "id_outlet"]].to_numpy())) == pares_base
        ant = df[df.periodo_demanda == "2023-07-01"]["demanda_unidades"].sum()
        assert 0.5 < novo["demanda_unidades"].sum() / ant < 2.5     # mesmo mes, um ano depois

    def test_rateio_semanal_alem_do_periodo_do_config(self, gerado):
        from datetime import date
        from src.sintetico.extensao import gerar_meses
        from src.sintetico.pipeline_semanal import desagregar_semanal
        _, _, _, est = gerado
        novo = gerar_meses(est, date(2024, 7, 1))
        sem = desagregar_semanal(novo)
        assert sem["demanda_unidades"].sum() == novo["demanda_unidades"].sum()
        assert sem["nr_semana"].max() >= 4


class TestOrquestradorEAgentes12:
    """Agentes 1, 2 e Desempate integrados pelo Orquestrador (motores substitutos)."""

    @pytest.fixture(scope="class")
    def corpus(self):
        import src.agentes  # noqa: F401
        from _schema import preparar_corpus
        textos = {
            "Alto": ["Anvisa aprova recall de lote de metformina por contaminacao",
                     "Falta de insulina nas farmacias preocupa pacientes, shortage confirmado",
                     "CMED autoriza reajuste de preco de medicamentos para diabetes"],
            "Medio": ["Estudo sugere beneficio cardiovascular da empagliflozina",
                      "Laboratorio anuncia parceria para pesquisa em obesidade",
                      "Dolar sobe e pressiona custo de importacao de IFA"],
            "Baixo": ["Campanha de conscientizacao sobre diabetes em escolas",
                      "Entrevista com medico sobre alimentacao saudavel",
                      "Prefeitura promove caminhada no parque"],
        }
        linhas = []
        for rot, lista in textos.items():
            for rep in range(4):
                for i, t in enumerate(lista):
                    linhas.append({"titulo": f"{t} ({rep}-{i})", "descricao": t, "conteudo": t,
                                   "url": f"u/{rot}/{rep}/{i}",
                                   "Avaliador 1": rot, "Avaliador 2": rot, "Avaliador 3": rot})
        linhas.append({"titulo": "caso dificil", "descricao": "x", "conteudo": "x", "url": "u/x",
                       "Avaliador 1": "Alto", "Avaliador 2": "Medio", "Avaliador 3": "Baixo"})
        df, _ = preparar_corpus(pd.DataFrame(linhas), verbose=False)
        return df

    def test_discordancia_total_fica_sem_rotulo(self, corpus):
        assert corpus.loc[corpus.titulo == "caso dificil", "classificacao_final"].isna().all()
        assert (corpus["situacao_anotacao"] == "DISCORDANCIA_TOTAL").sum() == 1

    def test_nb_tem_interface_do_agente1(self, corpus):
        from substitutos import ClassificadorNBImpacto
        ag = ClassificadorNBImpacto()
        rel = ag.treinar(corpus.dropna(subset=["classificacao_final"]), n_folds=3)
        assert 0 <= rel["f1_macro_media"] <= 1
        assert {"classe_predita", "score", "fold", "escopo_relevancia", "direcao_impacto"} <= set(ag.predicoes_oof_)
        assert (ag.predicoes_oof_["origem_predicao"] == "out_of_fold").all()
        novo = ag.classificar_dataframe(corpus.head(3))
        assert novo["score"].between(0, 1).all()

    def test_agente2_roda_com_encoder_lexical_e_exclui_auto_referencia(self, corpus):
        from agente2_validador_ajuste import AgenteValidadorAjuste
        from substitutos import ClassificadorNBImpacto, EncoderLexical
        rot = corpus.dropna(subset=["classificacao_final"])
        nb = ClassificadorNBImpacto()
        nb.treinar(rot, n_folds=3)
        enc = EncoderLexical().ajustar_idf(rot["texto_completo"].tolist())
        ag = AgenteValidadorAjuste(modelo_embedding="lexical", encoder=enc, cache_dir=None)
        ag.indexar_eventos_historicos(rot)
        val = ag.validar(nb.predicoes_oof_, salvar_relatorio=None)
        assert len(val) == len(nb.predicoes_oof_)                 # marca, nao apaga
        assert (val["similaridade_max"] < 0.9999).all()           # nunca se encontra no indice
        assert set(val["status_validacao"]) <= {"APROVADO", "DESCARTADO_BAIXO_IMPACTO", "DUPLICADO",
                                                "QUARENTENA_REVISAO_HUMANA", "SEM_ANCORA_DE_MERCADO"}

    def test_contrato_para_agente4(self, corpus):
        from agente2_validador_ajuste import AgenteValidadorAjuste
        from substitutos import ClassificadorNBImpacto, EncoderLexical
        from src.agentes.orquestrador import para_contrato_agente4
        rot = corpus.dropna(subset=["classificacao_final"])
        nb = ClassificadorNBImpacto(); nb.treinar(rot, n_folds=3)
        enc = EncoderLexical().ajustar_idf(rot["texto_completo"].tolist())
        ag = AgenteValidadorAjuste(modelo_embedding="lexical", encoder=enc, cache_dir=None)
        ag.indexar_eventos_historicos(rot)
        val = ag.validar(nb.classificar_dataframe(rot.head(9)), salvar_relatorio=None)
        out = para_contrato_agente4(val, {"agente1": "naive_bayes", "agente2": "encoder_lexical"})
        assert {"id_noticia", "nivel_impacto", "sentimento", "escopo", "status_validacao"} <= set(out)
        assert set(out["nivel_impacto"]) <= {"alto", "medio", "baixo"}
        assert set(out["sentimento"]) <= {"positivo", "neutro", "negativo"}
