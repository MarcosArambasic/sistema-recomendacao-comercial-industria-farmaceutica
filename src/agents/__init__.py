"""
Agente 3 - Comparador de Vendas (XGBoost)

Baseado na secao 3.4.3 do TCC. Cruza dados de vendas sinteticos com metas e
benchmarks, treina um XGBoost para prever o desvio percentual em relacao a
meta, classifica desvios por magnitude/tendencia, sinaliza risco de nao
atingimento de meta e usa SHAP para explicar quais atributos mais contribuem
para cada desvio.

Output: serie temporal estruturada (JSON/DataFrame) consumida pelo Agente 4
e pelo painel de insights final.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

try:
    import shap
    SHAP_DISPONIVEL = True
except ImportError:
    SHAP_DISPONIVEL = False


RiscoNivel = Literal["baixo", "medio", "alto"]


@dataclass
class DesvioClassificado:
    produto: str
    territorio: str
    representante: str
    data: str
    desvio_pct_real: float
    desvio_pct_previsto: float
    magnitude: RiscoNivel
    tendencia: Literal["queda", "estavel", "crescimento"]
    risco_meta: bool
    principais_fatores: list


class AgenteComparadorVendas:
    """Agente 3 - analise quantitativa de performance comercial via XGBoost."""

    FEATURES_NUMERICAS = [
        "meta_mil_r$",
        "market_share_relativo",
        "tendencia_prescricao",
        "mes",
    ]
    FEATURES_CATEGORICAS = ["produto", "territorio", "representante"]
    TARGET = "desvio_pct"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.encoders: dict[str, LabelEncoder] = {}
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Pre-processamento
    # ------------------------------------------------------------------
    def _preparar_features(self, df: pd.DataFrame, treino: bool) -> pd.DataFrame:
        df = df.copy()
        for col in self.FEATURES_CATEGORICAS:
            if treino:
                enc = LabelEncoder()
                df[f"{col}_enc"] = enc.fit_transform(df[col])
                self.encoders[col] = enc
            else:
                enc = self.encoders[col]
                # trata categorias nao vistas no treino
                df[f"{col}_enc"] = df[col].map(
                    lambda v: enc.transform([v])[0] if v in enc.classes_ else -1
                )
        return df

    def _matriz_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.FEATURES_NUMERICAS + [f"{c}_enc" for c in self.FEATURES_CATEGORICAS]
        return df[cols]

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def treinar(self, df: pd.DataFrame, test_size: float = 0.2) -> dict:
        df_prep = self._preparar_features(df, treino=True)
        X = self._matriz_features(df_prep)
        y = df_prep[self.TARGET]
        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Regularizacao L1/L2 conforme justificado no TCC (resistencia a overfitting)
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            reg_alpha=0.5,   # L1
            reg_lambda=1.0,  # L2
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = self.model.predict(X_test)
        metricas = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 3),
            "r2": round(float(r2_score(y_test, y_pred)), 3),
            "n_treino": len(X_train),
            "n_teste": len(X_test),
        }
        return metricas

    # ------------------------------------------------------------------
    # Inferencia / classificacao de desvios
    # ------------------------------------------------------------------
    @staticmethod
    def _classificar_magnitude(desvio_pct: float) -> RiscoNivel:
        abs_d = abs(desvio_pct)
        if abs_d < 10:
            return "baixo"
        elif abs_d < 25:
            return "medio"
        return "alto"

    @staticmethod
    def _classificar_tendencia(desvio_pct: float) -> str:
        if desvio_pct < -5:
            return "queda"
        elif desvio_pct > 5:
            return "crescimento"
        return "estavel"

    def _principais_fatores(self, X_row: pd.DataFrame, top_n: int = 3) -> list[str]:
        if not SHAP_DISPONIVEL or self.model is None:
            return []
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_row)
        contribs = list(zip(X_row.columns, shap_values[0]))
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        return [nome.replace("_enc", "") for nome, _ in contribs[:top_n]]

    def analisar(self, df: pd.DataFrame, antecedencia_dias: int = 30) -> list[DesvioClassificado]:
        """
        Analisa desvios por produto/territorio/representante, classifica-os
        por magnitude e tendencia, e sinaliza risco de nao atingimento de
        meta com antecedencia configuravel (parametro simbolico aqui, pois
        o dado sintetico ja e mensal consolidado).
        """
        if self.model is None:
            raise RuntimeError("Modelo nao treinado. Chame treinar() primeiro.")

        df_prep = self._preparar_features(df, treino=False)
        X = self._matriz_features(df_prep)
        previsoes = self.model.predict(X)

        resultados = []
        for i, (_, row) in enumerate(df_prep.iterrows()):
            desvio_real = row[self.TARGET]
            desvio_prev = float(previsoes[i])
            magnitude = self._classificar_magnitude(desvio_real)
            tendencia = self._classificar_tendencia(desvio_real)
            risco = magnitude in ("medio", "alto") and tendencia == "queda"

            fatores = []
            if SHAP_DISPONIVEL and i < 200:  # limitar custo computacional
                fatores = self._principais_fatores(X.iloc[[i]])

            resultados.append(
                DesvioClassificado(
                    produto=row["produto"],
                    territorio=row["territorio"],
                    representante=row["representante"],
                    data=row["data"],
                    desvio_pct_real=round(float(desvio_real), 2),
                    desvio_pct_previsto=round(desvio_prev, 2),
                    magnitude=magnitude,
                    tendencia=tendencia,
                    risco_meta=risco,
                    principais_fatores=fatores,
                )
            )
        return resultados

    def exportar_serie_temporal(self, resultados: list[DesvioClassificado]) -> pd.DataFrame:
        """Estrutura os resultados em serie temporal para o Agente 4 e o painel final."""
        return pd.DataFrame([asdict(r) for r in resultados])


if __name__ == "__main__":
    # BASE_DIR aponta para a raiz do projeto (2 niveis acima de src/agents/),
    # entao o script roda corretamente independente do diretorio de trabalho.
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    df = pd.read_csv(BASE_DIR / "data" / "vendas_sinteticas.csv")

    agente3 = AgenteComparadorVendas()
    metricas = agente3.treinar(df)
    print("Metricas de treino do Agente 3 (XGBoost):")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    # analisa uma amostra recente (ultimo mes disponivel) para nao explodir custo
    ultimo_mes = df["data"].max()
    amostra = df[df["data"] == ultimo_mes]
    resultados = agente3.analisar(amostra)

    df_out = agente3.exportar_serie_temporal(resultados)
    out_dir = BASE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_dir / "agente3_desvios.csv", index=False)

    print(f"\n{len(df_out)} registros analisados no periodo {ultimo_mes}")
    print(f"Casos de risco de meta sinalizados: {df_out['risco_meta'].sum()}")
    print("\nAmostra de saida:")
    print(df_out.head(5).to_string(index=False))
