import pandas as pd
import seaborn as sns
from typing import Tuple
import matplotlib.pyplot as plt
from ensure import ensure_annotations

from sales_project.utils import get_bins

sns.set_theme(context="talk", style="darkgrid", palette="dark", font="sans-serif")


# @ensure_annotations
def linear_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str | None = None,
    y_label: str | None = None,
    hue: str | None = None,
    figsize: Tuple[int, int] = (14, 5),
    use_index: bool = False,
    scatter: bool = False,
    linear: bool = True,
    y_scale: str | None = None,
    x_scale: str | None = None,
) -> None:
    """
    Plots a linear plot with optional scatter points for the given data.

    Args:
        data (pd.DataFrame):
            The input data frame.
        x (str):
            The column name for the x-axis.
        y (str):
            The column name for the y-axis.
        title (str):
            The title of the plot.
        hue (str, optional):
            The column name to use for coloring the lines/points.
        figsize (Tuple[int, int], optional):
            The figure size for the plot.
        use_index (bool, optional):
            Whether to use the index of the data frame for the x-axis.
        scatter (bool, optional):
            Whether to include scatter points on the plot.
    """
    plt.figure(figsize=figsize)
    if use_index:
        if linear:
            sns.lineplot(x=data.index, y=data[y], hue=data[hue] if hue else None)
        if scatter:
            sns.scatterplot(x=data.index, y=data[y], hue=data[hue] if hue else None)
    else:
        if linear:
            sns.lineplot(data=data, x=x, y=y, hue=hue if hue else None)
        if scatter:
            sns.scatterplot(data=data, x=x, y=y, hue=hue if hue else None)
    plt.title(title)
    plt.xlabel(x if x_label is None else x_label)
    plt.ylabel(y if y_label is None else y_label)
    if x_scale:
        plt.xscale(x_scale)
    if y_scale:
        plt.yscale(y_scale)
    plt.show()


@ensure_annotations
def hist_box_plot(
    df: pd.DataFrame, feature: str, kde: bool = False, label: str | None = None
) -> None:
    """
    Plots a histogram and box plot for a given feature in a DataFrame.

    Args:
        df (pd.DataFrame):
            The input DataFrame.
        feature (str):
            The column name of the feature to plot.
        kde (bool, optional):
            Whether to plot a kernel density estimate on the histogram.
            Defaults to False.
        label (str, optional):
            The label to use for the feature.
            If not provided, the feature name will be used.
    """
    if label == None:
        label = feature
    data = df[feature].dropna()
    plt.figure(figsize=(15, 6))
    sns.histplot(data, bins=get_bins(len(data)), kde=kde)
    plt.xlabel(label)
    plt.figure(figsize=(15, 3))
    sns.boxplot(data, orient="h")
    plt.xlabel(label)


@ensure_annotations
def train_submission_countplot(
    df: pd.DataFrame,
    col: str,
    is_submission_col: str,
) -> None:
    """
    Plots a count plot with three subplots: one for all data,
    one for train data, and one for submission data.

    Args:
        df (pd.DataFrame):
            The input DataFrame.
        col (str):
            The column name to plot the count plot for.
        is_submission_col (str):
            The column name that indicates if a row is in the
            submission data.
    """

    fig, ax = plt.subplots(2, 1, figsize=(10, 2 * 10 / 24 * df[col].nunique()))
    sns.countplot(
        df[col],
        order=sorted(df[col].unique()),
        ax=ax[0],
        stat="percent",
        label="All data",
    )
    ax[0].legend()
    sns.countplot(
        df.query(f"{is_submission_col} == False")[col],
        order=sorted(df.query(f"{is_submission_col} == False")[col].unique()),
        ax=ax[1],
        color="blue",
        alpha=0.5,
        stat="percent",
        label="Train data",
    )
    sns.countplot(
        df.query(f"{is_submission_col} == True")[col],
        order=sorted(df.query(f"{is_submission_col} == True")[col].unique()),
        ax=ax[1],
        color="red",
        alpha=0.5,
        stat="percent",
        label="Submission data",
    )
    ax[1].legend()
    fig.tight_layout()
