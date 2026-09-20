"""
=======
Carga de datos crudos del dataset SkillCorner (Liga Endesa ACB 2025/2026)
para el reto de baloncesto del SkillCorner x PySport Analytics Cup 2.0.

Este módulo NO transforma ni limpia datos: solo los lee desde
`data/01_raw/` y los devuelve como estructuras de Python (dict) o
DataFrames de pandas. La limpieza y las features derivadas viven en
`features.py`.

Convenciones:
- Cada partido tiene un `game_id` (int) que identifica su carpeta.
- Las tablas de eventos (shots, picks, drives, etc.) están dentro de
  `{game_id}_dynamic_events.json`, una clave por tabla.
"""

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------

# Carpeta donde copiamos, sin modificar, los datos originales de SkillCorner.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpeta donde copiamos, sin modificar, los datos originales de SkillCorner.
RAW_DIR = PROJECT_ROOT / "data" / "01_raw" / "skillcorner"
MATCHES_DIR = RAW_DIR / "matches"
AGGREGATES_DIR = RAW_DIR / "aggregates"

# Los 10 partidos de muestra incluidos en el dataset (con eventos completos;
# el tracking crudo puede o no estar descargado vía Git LFS para cada uno).
SAMPLE_GAME_IDS: list[int] = [
    114086,
    114099,
    114169,
    114234,
    114243,
    178442,
    179612,
    184439,
    188630,
    191313,
]


# ---------------------------------------------------------------------------
# Carga por partido individual
# ---------------------------------------------------------------------------


def load_game_data(game_id: int) -> dict:
    """Carga los metadatos de un partido (rosters, equipos, fecha).

    Parameters
    ----------
    game_id : int
        Identificador del partido, igual al nombre de su carpeta en
        `data/matches/`.

    Returns
    -------
    dict
        Contenido crudo de `{game_id}_game_data.json`.
    """
    path = MATCHES_DIR / str(game_id) / f"{game_id}_game_data.json"

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_dynamic_events(game_id: int) -> dict:
    """Carga el diccionario completo de eventos dinámicos de un partido.

    El archivo original agrupa ~20 tablas distintas (shots, picks,
    drives, closeouts, chances, matchups, etc.) bajo una sola clave
    por tabla.

    Parameters
    ----------
    game_id : int
        Identificador del partido.

    Returns
    -------
    dict
        Mapeo {nombre_de_tabla: lista_de_eventos}, tal como viene en
        `{game_id}_dynamic_events.json`.
    """
    path = MATCHES_DIR / str(game_id) / f"{game_id}_dynamic_events.json"

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_table(game_id: int, table_name: str) -> pd.DataFrame:
    """Carga una sola tabla de eventos de un partido como DataFrame.

    Parameters
    ----------
    game_id : int
        Identificador del partido.
    table_name : str
        Nombre de la tabla dentro de `dynamic_events.json`
        (ej. "shots", "picks", "drives", "closeouts", "chances").

    Returns
    -------
    pd.DataFrame
        Una fila por evento, con una columna adicional `game_id` para
        poder identificar el origen una vez se unan varios partidos.

    Raises
    ------
    KeyError
        Si `table_name` no existe en los eventos de ese partido.
    """
    events = load_dynamic_events(game_id)

    if table_name not in events:
        available = ", ".join(sorted(events.keys()))
        raise KeyError(
            f"'{table_name}' no existe en el partido {game_id}. "
            f"Tablas disponibles: {available}"
        )

    table_df = pd.DataFrame(events[table_name])
    table_df["game_id"] = game_id

    return table_df


# ---------------------------------------------------------------------------
# Carga combinada (varios partidos)
# ---------------------------------------------------------------------------


def load_all_games(
    table_name: str,
    game_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Une la misma tabla de eventos de varios partidos en un DataFrame.

    Parameters
    ----------
    table_name : str
        Nombre de la tabla a cargar (ej. "shots").
    game_ids : list[int], optional
        Partidos a incluir. Si no se especifica, se usan los 10
        partidos de muestra definidos en `SAMPLE_GAME_IDS`.

    Returns
    -------
    pd.DataFrame
        Todas las filas concatenadas, con la columna `game_id`
        indicando la procedencia de cada evento.
    """
    if game_ids is None:
        game_ids = SAMPLE_GAME_IDS

    tables = [load_table(game_id, table_name) for game_id in game_ids]

    return pd.concat(tables, ignore_index=True)


# ---------------------------------------------------------------------------
# Agregados de temporada completa (293 de 327 partidos, solo ofensivo)
# ---------------------------------------------------------------------------


def load_season_aggregate(metric: str) -> pd.DataFrame:
    """Carga uno de los 3 CSV de agregados de temporada.

    Estos agregados cubren 293 de los 327 partidos de la temporada
    completa, pero solo incluyen métricas ofensivas (no hay agregados
    defensivos a nivel de temporada).

    Parameters
    ----------
    metric : str
        Uno de: "shots", "drives", "picks".

    Returns
    -------
    pd.DataFrame
        Contenido del CSV correspondiente.

    Raises
    ------
    ValueError
        Si `metric` no es uno de los valores permitidos.
    """
    valid_metrics = {"shots", "drives", "picks"}

    if metric not in valid_metrics:
        raise ValueError(
            f"metric debe ser uno de {valid_metrics}, se recibió '{metric}'"
        )

    path = AGGREGATES_DIR / f"acb_{metric}aggregates_20252026.csv"

    return pd.read_csv(path)