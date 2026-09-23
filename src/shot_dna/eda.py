"""
eda.py
======
Utilidades de exploración rápida (EDA) reutilizables para cualquier
tabla del dataset SkillCorner, sin repetir el mismo bloque de código
por cada una de las 20 tablas.
"""

import pandas as pd


def profile_table(df: pd.DataFrame, name: str = "tabla") -> None:
    """Imprime un resumen estándar de una tabla.

    Incluye forma, nulos, estadísticas numéricas y la distribución de
    las columnas categóricas con pocas categorías.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla a perfilar (ej. resultado de `load_all_games("shots")`).
    name : str
        Nombre descriptivo de la tabla, solo para el encabezado impreso.
    """
    print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")
    print(f"Forma: {df.shape[0]} filas x {df.shape[1]} columnas\n")

    print("--- Nulos por columna (solo columnas con nulos) ---")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    print(nulls if not nulls.empty else "Sin nulos.")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        print("\n--- Estadísticas numéricas ---")
        print(df[numeric_cols].describe().T)

    categorical_cols = df.select_dtypes(include=["object", "bool", "string"]).columns
    for col in categorical_cols:
        n_unique = df[col].nunique()
        # Solo mostramos el detalle si son pocas categorías; columnas
        # tipo Id con miles de valores únicos no aportan aquí.
        if 1 < n_unique <= 20:
            print(f"\n--- {col} ({n_unique} valores únicos) ---")
            print(df[col].value_counts(dropna=False))


def effect_size_table(
    df: pd.DataFrame,
    target: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    min_n: int = 20,
) -> pd.DataFrame:
    """Ranking de qué tanto se asocia cada variable con el target.

    - Numéricas: correlación de Spearman (ρ). Es robusta a outliers y
      capta relaciones monótonas aunque no sean lineales.
    - Categóricas: eta cuadrado (η²), la proporción de la varianza del
      target que explican las diferencias entre categorías.

    Para poder ordenarlas juntas se reporta la columna `fuerza`:
    ρ² para numéricas y η² para categóricas. Ambas aproximan "qué
    porcentaje de la variación del target se asocia con la variable".
    Es una comparación orientativa, no una prueba formal, y describe
    asociación, no causalidad.

    Parameters
    ----------
    df : pd.DataFrame
        Datos con el target y las variables a evaluar.
    target : str
        Variable objetivo numérica (ej. "shotQuality").
    numeric_cols : list[str]
        Variables numéricas a evaluar.
    categorical_cols : list[str]
        Variables categóricas o booleanas a evaluar.
    min_n : int, optional
        Tamaño mínimo por categoría para incluirla en el cálculo de η².

    Returns
    -------
    pd.DataFrame
        Una fila por variable, ordenada de mayor a menor fuerza.
    """
    rows = []

    for col in numeric_cols:
        data = df[[col, target]].dropna()
        rho = data[col].astype(float).corr(data[target], method="spearman")
        rows.append(
            {
                "variable": col,
                "tipo": "numérica",
                "métrica": "Spearman ρ",
                "valor": rho,
                "fuerza": rho**2,
                "n": len(data),
            }
        )

    for col in categorical_cols:
        data = df[[col, target]].dropna()

        # Excluimos categorías pequeñas para no inflar η² con ruido.
        counts = data[col].value_counts()
        data = data[data[col].isin(counts[counts >= min_n].index)]

        grand_mean = data[target].mean()
        ss_total = ((data[target] - grand_mean) ** 2).sum()
        group_stats = data.groupby(col, observed=True)[target].agg(["mean", "size"])
        ss_between = (
            group_stats["size"] * (group_stats["mean"] - grand_mean) ** 2
        ).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else float("nan")

        rows.append(
            {
                "variable": col,
                "tipo": "categórica",
                "métrica": "η²",
                "valor": eta_squared,
                "fuerza": eta_squared,
                "n": len(data),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("fuerza", ascending=False)
        .reset_index(drop=True)
    )