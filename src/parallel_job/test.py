from __future__ import annotations

from typing import TYPE_CHECKING

from .doc import append_docstring, doc_test_parallel
from .parallel import parallel_wrapper
from .utils import num_string

if TYPE_CHECKING:
    from typing import Any, Callable

    from .type_helper import ArgsList, KwargsList


@append_docstring(doc_test_parallel)
def test_parallel(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList | None = None,
    n_jobs: int = 1,
    verbose: int = 0,
    backends: list[str] | None = None,
    name: str = "",
    log_level: int = 20,
    fig_type: str = "jpg",
) -> None:
    """
    Test parallel execution using different backends and visualize the results.
    """
    import logging

    logging.basicConfig(level=log_level, format="%(message)s")
    log = logging.getLogger("parallel")

    if backends is None:
        backends = [
            "for_loop",
            "list_comprehension",
            "async_run",
            "threading_run",
            "pool",
            "process",
            "thread_pool_executor",
            "process_pool_executor",
            "joblib_loky",
            "joblib_threading",
            "subinterpreter_json_pickle",
            "subinterpreter_pickle_pickle",
        ]

    name = name or func.__name__
    log.info(f"testing {name}\n---")

    ret = {}
    for backend in backends:
        ret[backend] = parallel_wrapper(
            func,
            args_list=args_list,
            kwargs_list=kwargs_list,
            n_jobs=n_jobs,
            verbose=verbose,
            backend=backend,
            measure_time=True,
        )
        log.info(
            f"{backend:25s}: {num_string((ret[backend][1][1] - ret[backend][1][0]).total_seconds())}s"
        )
    ref = [x[0] for x in list(ret.values())[0][0]]
    for x in ret:
        v = [x[0] for x in ret[x][0]]
        if v != ref:
            log.info(f"{x} is different from ref")
            log.info(v)
            log.info(ref)

    max_time = max([(x[1][1] - x[1][0]).total_seconds() for x in ret.values()])
    from .chart import waterfall_chart

    for backend in ret:
        waterfall_chart(
            f"{name}_{backend}",
            ret[backend][1][0],
            ret[backend][1][1],
            [x[1] for x in ret[backend][0]],
            max_time,
            fig_type,
        )

    names = list(ret)
    width = [(x[1][1] - x[1][0]).total_seconds() for x in ret.values()]
    from .chart import summary_chart

    summary_chart(f"{name}_summary", names, width, fig_type)


def sleep_1(x: int) -> int:
    """
    Sleep 1 seconds and return input value.

    Parameters
    ----------
    x : int
        Input value.

    Returns
    -------
    int
        The input value.
    """
    import time

    time.sleep(1)
    return x


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

    time.sleep(x)
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
