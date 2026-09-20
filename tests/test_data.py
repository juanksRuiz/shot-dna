"""
test_data.py
============
Pruebas para las funciones de carga en shot_dna/data.py.

Se ejecutan con: uv run pytest
"""

import pandas as pd

from shot_dna.data import (
    SAMPLE_GAME_IDS,
    load_all_games,
    load_dynamic_events,
    load_game_data,
    load_table,
)

# Usamos un solo partido conocido para pruebas rápidas.
TEST_GAME_ID = SAMPLE_GAME_IDS[0]


def test_load_game_data_returns_dict():
    """load_game_data debe devolver un diccionario con datos del partido."""
    game = load_game_data(TEST_GAME_ID)
    assert isinstance(game, dict)


def test_load_dynamic_events_contains_shots_table():
    """El dict de eventos debe incluir la tabla 'shots'."""
    events = load_dynamic_events(TEST_GAME_ID)
    assert "shots" in events


def test_load_table_returns_dataframe_with_game_id():
    """load_table debe devolver un DataFrame con la columna game_id."""
    shots = load_table(TEST_GAME_ID, "shots")
    assert isinstance(shots, pd.DataFrame)
    assert "game_id" in shots.columns
    assert (shots["game_id"] == TEST_GAME_ID).all()


def test_load_table_invalid_name_raises_keyerror():
    """Pedir una tabla que no existe debe lanzar KeyError, no fallar en silencio."""
    try:
        load_table(TEST_GAME_ID, "tabla_que_no_existe")
        assert False, "Debió lanzar KeyError"
    except KeyError:
        pass


def test_load_all_games_concatenates_multiple_matches():
    """load_all_games debe traer filas de más de un partido."""
    two_games = SAMPLE_GAME_IDS[:2]
    shots = load_all_games("shots", game_ids=two_games)
    assert set(shots["game_id"].unique()) == set(two_games)