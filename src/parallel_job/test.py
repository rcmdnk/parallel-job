from __future__ import annotations

from typing import TYPE_CHECKING

from .parallel import parallel_wrapper
from .utils import num_string

if TYPE_CHECKING:
    from typing import Any, Callable


def test_parallel(
    func: Callable[..., Any],
    name: str | None = None,
    n_jobs: int = 1,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    fig_type: str = "jpg",
) -> None:
    """
    Test parallel execution using different backends and visualize the results.

    Parameters
    ----------
    func : Callable
        The function to execute.
    name : str, optional
        The name of the test (default is None).
    n_jobs : int, optional
        Number of parallel jobs (default is 1).
    args : tuple, optional
        The positional arguments for the function (default is ()).
    kwargs : dict | None, optional
        The keyword arguments for the function (default is None).
    fig_type : str, optional
        File type for the saved chart (default is "jpg").
    """
    import logging

    log = logging.getLogger("parallel")

    name = name or func.__name__
    log.info(f"testing {name}\n---")

    ret = {}
    for backend in [
        "for_loop",
        "list_comprehension",
        "async_job",
        "threading_job",
        "pool",
        "process",
        "thread_pool_executor",
        "process_pool_executor",
        "joblib_loky",
        "joblib_threading",
        "subinterpreter",
    ]:
        ret[backend] = parallel_wrapper(
            func,
            n_jobs=n_jobs,
            args=args,
            kwargs=kwargs,
            backend=backend,
            verbose=0,
            measure_time=True,
        )
        log.info(
            f"{backend:25s}: {num_string((ret[backend][1][1] - ret[backend][1][0]).total_seconds())}s"
        )
    ref = list(ret.values())[0][0][0]
    for x in ret:
        if ret[x][0] != ref:
            log.info(f"{x} is different from ref")
            log.info(ret[x][0], ref)

    max_time = max([(x[2] - x[1]).total_seconds() for x in ret.values()])
    from .chart import waterfall_chart

    for backend in ret:
        waterfall_chart(
            f"{name}_{backend}",
            ret[backend][1],
            ret[backend][2],
            ret[backend][3],
            max_time,
            fig_type,
        )

    width = [(x[2] - x[1]).total_seconds() for x in ret.values()]
    names = list(ret)
    from .chart import summary_chart

    summary_chart(f"{name}_summary", names, width, fig_type)


def sleep(x: int) -> int:
    """
    Sleep for a given number of seconds.

    Parameters
    ----------
    x : int
        Number of seconds to sleep.

    Returns
    -------
    int
        The input value.
    """
    import time

    time.sleep(1)
    return x


def n_prime(start: int, end: int) -> int:
    """
    Count the number of prime numbers in a given range.

    Parameters
    ----------
    start : int
        The start of the range.
    end : int
        The end of the range.

    Returns
    -------
    int
        Number of prime numbers in the range.
    """
    n = 0
    for i in range(start, end + 1):
        if i <= 1:
            continue
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:  # noqa: S001
                continue
        n += 1
    return n
