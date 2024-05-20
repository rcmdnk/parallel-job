from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from datetime import datetime

from .utils import num_string


def waterfall_chart(
    name: str,
    start: datetime,
    end: datetime,
    starts_ends: list[tuple[datetime, datetime]],
    max_time: float,
    fig_type: str = "jpg",
) -> None:
    """
    Create a waterfall chart to visualize task durations.

    Parameters
    ----------
    name : str
        The name of the chart.
    start : datetime
        The start time of the entire process.
    end : datetime
        The end time of the entire process.
    starts_ends : list of tuple of datetime
        List of start and end times for each task.
    max_time : float
        Maximum time value for the x-axis.
    fig_type : str, optional
        File type for the saved chart (default is "jpg").
    """
    data = [
        ((x[0] - start).total_seconds(), (x[1] - x[0]).total_seconds())
        for x in reversed(starts_ends)
    ]
    starts, lengthes = zip(*data)
    names = [f"job_{str(i)}" for i in (range(len(data), 0, -1))]
    _, ax = plt.subplots(figsize=(6, len(data) * 0.5), layout="tight")
    bars = ax.barh(
        y=range(len(data)),
        tick_label=names,
        width=lengthes,
        left=starts,
        height=1,
        color="skyblue",
        edgecolor="black",
    )

    for bar in bars:
        ax.text(
            max_time * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"[{num_string(bar.get_x())}s, {num_string(bar.get_width())}s]",
            ha="left",
            va="center",
            color="black",
        )

    ax.set_xlim(0, max_time)
    end_line_x = (end - start).total_seconds()
    ax.axvline(x=end_line_x, linestyle="--", color="indianred")
    ylim = ax.get_ylim()
    ax.text(
        end_line_x + max_time * 0.02,
        ylim[1] - (ylim[1] - ylim[0]) * 0.1,
        "end",
        ha="center",
        va="bottom",
        rotation="vertical",
        color="indianred",
    )
    ax.set_ylabel("job")
    ax.set_xlabel("time [s]")
    ax.set_title(name)
    plt.subplots_adjust(right=0.95)
    plt.savefig(f"{name}.{fig_type}")
    plt.close()


def summary_chart(
    name: str, backends: list[str], lengths: list[float], fig_type: str = "jpg"
) -> None:
    _, ax = plt.subplots(
        figsize=(6, (len(backends) + 3) * 0.5), layout="tight"
    )
    bars = ax.barh(
        y=backends,
        width=lengths,
        height=1,
        color="skyblue",
        edgecolor="black",
    )

    for bar in bars:
        ax.text(
            bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f"{num_string(bar.get_width())}",
            ha="center",
            va="center",
            color="black",
        )

    ax.set_xlabel("time [s]")
    ax.set_title(name)
    plt.savefig(f"{name}.{fig_type}")
    plt.close()
