"""
viz.py
======
Funciones de visualización reutilizables para el EDA de shot-dna.

Todas las funciones reciben un DataFrame y devuelven el objeto `Axes`
de matplotlib, para poder combinarlas en subplots o ajustarlas después.
La cancha se dibuja con mplbasketball usando la misma convención de
coordenadas que SkillCorner (pies, origen en el centro, reglas FIBA).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mplbasketball import Court

# Convención del dataset (ver docs/PRIMER.md de SkillCorner):
# unidades en pies, origen (0, 0) en el centro, aro ofensivo en x negativo.
COURT_KWARGS = {"court_type": "fiba", "origin": "center", "units": "ft"}

# Variable objetivo por defecto en todo el proyecto.
TARGET = "shotQuality"


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _annotate_counts(ax, counts):
    """Escribe el tamaño de muestra (n) en la parte baja de cada categoría.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje donde se escribe.
    counts : list[int]
        Tamaño de muestra de cada categoría, en el orden del eje x.
    """
    y_bottom = ax.get_ylim()[0]
    for position, n in enumerate(counts):
        ax.text(
            position,
            y_bottom,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
        )


# ---------------------------------------------------------------------------
# Distribución del target por categoría
# ---------------------------------------------------------------------------


def plot_box_by_category(
    df,
    category,
    target=TARGET,
    min_n=20,
    order=None,
    palette="Spectral",
    ax=None,
    title=None,
):
    """Boxplot coloreado del target por categoría, con n por grupo.

    Las categorías con menos de `min_n` observaciones se excluyen, porque
    su mediana no es estadísticamente confiable. Se imprimen para que
    quede registro de qué se dejó fuera.

    Parameters
    ----------
    df : pd.DataFrame
        Datos con la columna categórica y el target.
    category : str
        Columna categórica a comparar.
    target : str, optional
        Variable numérica a graficar (por defecto `shotQuality`).
    min_n : int, optional
        Tamaño mínimo de muestra para incluir una categoría.
    order : list, optional
        Orden fijo de categorías (ej. nivel de contest de menor a mayor).
        Si no se da, se ordenan por mediana descendente.
    palette : str, optional
        Paleta de colores de seaborn/matplotlib.
    ax : matplotlib.axes.Axes, optional
        Eje donde dibujar. Si no se da, se crea uno nuevo.
    title : str, optional
        Título del gráfico.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    # Filtramos categorías con poca muestra y avisamos cuáles quedaron fuera.
    counts = df[category].value_counts()
    excluded = counts[counts < min_n]
    if not excluded.empty:
        print(f"[{category}] excluidas por n < {min_n}: {excluded.to_dict()}")

    data = df[df[category].isin(counts[counts >= min_n].index)].copy()
    data[category] = data[category].astype(str)

    if order is None:
        order = (
            data.groupby(category)[target]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
    else:
        present = set(data[category].unique())
        order = [str(c) for c in order if str(c) in present]

    sns.boxplot(
        data=data,
        x=category,
        y=target,
        order=order,
        hue=category,
        hue_order=order,
        palette=palette,
        legend=False,
        showfliers=False,
        ax=ax,
    )
    # Puntos individuales semitransparentes: muestran densidad real.
    sns.stripplot(
        data=data,
        x=category,
        y=target,
        order=order,
        color="black",
        size=2,
        alpha=0.25,
        ax=ax,
    )
    ax.axhline(
        df[target].median(),
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Mediana global",
    )

    _annotate_counts(ax, [(data[category] == c).sum() for c in order])

    ax.set_title(title or f"{target} por {category}")
    ax.set_xlabel(category)
    ax.set_ylabel(target)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper right")

    return ax


def plot_binned_mean(
    df,
    column,
    bins,
    labels=None,
    target=TARGET,
    color="tab:purple",
    ax=None,
    title=None,
):
    """Media del target (con IC 95%) por rangos de una variable numérica.

    Útil para ver relaciones no lineales que una correlación simple
    esconde (ej. el efecto de la distancia al defensor puede saturarse).

    Parameters
    ----------
    df : pd.DataFrame
        Datos con la columna numérica y el target.
    column : str
        Variable numérica a discretizar.
    bins : list[float]
        Bordes de los rangos (se pasan a `pd.cut`).
    labels : list[str], optional
        Etiquetas legibles para cada rango.
    target : str, optional
        Variable numérica a promediar.
    color : str, optional
        Color de la línea y los puntos.
    ax : matplotlib.axes.Axes, optional
        Eje donde dibujar.
    title : str, optional
        Título del gráfico.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    data = df[[column, target]].dropna().copy()
    data["bin"] = pd.cut(
        data[column], bins=bins, labels=labels, include_lowest=True
    )

    sns.pointplot(
        data=data,
        x="bin",
        y=target,
        errorbar=("ci", 95),
        color=color,
        ax=ax,
    )

    counts = data["bin"].value_counts(sort=False).tolist()
    _annotate_counts(ax, counts)

    ax.set_title(title or f"{target} medio por rangos de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel(f"{target} medio (IC 95%)")

    return ax


# ---------------------------------------------------------------------------
# Validación de la variable objetivo
# ---------------------------------------------------------------------------


def plot_calibration(df, target=TARGET, outcome="outcome", n_bins=10, ax=None):
    """Compara el target (0-100) contra el % real de acierto por deciles.

    Si los puntos caen cerca de la diagonal, el target se comporta como
    una probabilidad de acierto esperada (tipo "expected FG%").

    Parameters
    ----------
    df : pd.DataFrame
        Tiros con el target y el resultado (True = anotado).
    target : str, optional
        Puntuación de calidad en escala 0-100.
    outcome : str, optional
        Columna booleana de acierto.
    n_bins : int, optional
        Número de grupos de igual tamaño (cuantiles).
    ax : matplotlib.axes.Axes, optional
        Eje donde dibujar.

    Returns
    -------
    tuple[matplotlib.axes.Axes, pd.DataFrame]
        El eje y la tabla de calibración (esperado vs. real por grupo).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    data = df[[target, outcome]].dropna().copy()
    data["bin"] = pd.qcut(data[target], q=n_bins, duplicates="drop")

    calibration = data.groupby("bin", observed=True).agg(
        expected=(target, "mean"),
        actual=(outcome, lambda s: 100 * s.mean()),
        n=(outcome, "size"),
    )

    ax.plot([0, 100], [0, 100], "--", color="gray", label="Calibración perfecta")
    ax.plot(calibration["expected"], calibration["actual"], color="tab:blue", alpha=0.6)
    scatter = ax.scatter(
        calibration["expected"],
        calibration["actual"],
        s=calibration["n"] * 2,
        c=calibration["expected"],
        cmap="viridis",
        edgecolor="black",
        zorder=3,
    )
    plt.colorbar(scatter, ax=ax, label=f"{target} medio del grupo")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"{target} medio (esperado)")
    ax.set_ylabel("% de acierto real")
    ax.set_title(f"Calibración: ¿{target} predice el acierto?")
    ax.legend(loc="upper left")

    return ax, calibration


# ---------------------------------------------------------------------------
# Gráficos sobre la cancha (mplbasketball)
# ---------------------------------------------------------------------------


def draw_half_court(ax=None):
    """Dibuja media cancha FIBA con el aro ofensivo (lado x negativo).

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Eje donde dibujar. Si no se da, se crea uno nuevo.

    Returns
    -------
    matplotlib.axes.Axes
    """
    court = Court(**COURT_KWARGS)
    if ax is None:
        _, ax = court.draw(orientation="hl")
    else:
        court.draw(ax=ax, orientation="hl")
    return ax


def plot_shot_chart(
    df,
    color_col=TARGET,
    categorical=False,
    cmap="plasma",
    ax=None,
    title=None,
):
    """Mapa de tiros sobre media cancha, coloreado por una variable.

    Parameters
    ----------
    df : pd.DataFrame
        Tiros con columnas `x` y `y` (en pies).
    color_col : str, optional
        Columna para colorear cada punto.
    categorical : bool, optional
        True si `color_col` es categórica (usa leyenda en vez de barra).
    cmap : str, optional
        Mapa de color para variables numéricas.
    ax : matplotlib.axes.Axes, optional
        Eje donde dibujar.
    title : str, optional
        Título del gráfico.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = draw_half_court(ax)
    data = df.dropna(subset=["x", "y", color_col])

    if categorical:
        sns.scatterplot(
            data=data,
            x="x",
            y="y",
            hue=color_col,
            palette="tab10",
            s=14,
            alpha=0.8,
            ax=ax,
            zorder=3,
        )
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        scatter = ax.scatter(
            data["x"],
            data["y"],
            c=data[color_col],
            cmap=cmap,
            vmin=0,
            vmax=100,
            s=14,
            alpha=0.8,
            zorder=3,
        )
        plt.colorbar(scatter, ax=ax, label=color_col, shrink=0.8)

    ax.set_title(title or f"Mapa de tiros coloreado por {color_col}")
    return ax


def plot_hex_quality(
    df,
    target=TARGET,
    gridsize=(16, 12),
    mincnt=5,
    cmap="RdYlGn",
    ax=None,
    title=None,
):
    """Mapa hexagonal del target medio por zona de la cancha.

    Cada hexágono muestra el promedio del target de los tiros que caen
    en él; se ocultan los hexágonos con menos de `mincnt` tiros.

    Parameters
    ----------
    df : pd.DataFrame
        Tiros con columnas `x`, `y` y el target.
    target : str, optional
        Variable a promediar por hexágono.
    gridsize : tuple[int, int], optional
        Resolución de la malla hexagonal.
    mincnt : int, optional
        Mínimo de tiros por hexágono para mostrarlo.
    cmap : str, optional
        Mapa de color (rojo = baja calidad, verde = alta).
    ax : matplotlib.axes.Axes, optional
        Eje donde dibujar.
    title : str, optional
        Título del gráfico.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = draw_half_court(ax)
    data = df.dropna(subset=["x", "y", target])

    hexbin = ax.hexbin(
        data["x"],
        data["y"],
        C=data[target],
        reduce_C_function=np.mean,
        gridsize=gridsize,
        mincnt=mincnt,
        extent=(-46, 0, -25, 25),
        cmap=cmap,
        alpha=0.85,
        zorder=0,
    )
    plt.colorbar(hexbin, ax=ax, label=f"{target} medio", shrink=0.8)

    ax.set_title(title or f"{target} medio por zona (mín. {mincnt} tiros)")
    return ax