from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable


def get_indent(doc: str) -> int:
    lines = doc.split("\n")
    if len(lines) < 2:
        return 0

    first_line = lines[1]
    indent = len(first_line) - len(first_line.lstrip())
    return indent


def add_indent(doc: str, indent: int) -> str:
    lines = doc.split("\n")
    return "\n".join(
        " " * indent + line if line.strip() != "" else "" for line in lines
    )


def append_docstring(doc: str) -> Callable[..., Any]:
    def _deco(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        if wrapper.__doc__ is None:
            return wrapper
        indent = get_indent(wrapper.__doc__)
        if indent == 0:
            return wrapper
        wrapper.__doc__ += add_indent(doc, indent)
        return wrapper

    return _deco


docs = {
    "func": """func : Callable[..., Any]
    The function to execute.""",
    "args_list": """args_list : Sized[tuple[Any]]
    The positional arguments for the function. Each task should have its
    own tuple of arguments.""",
    "kwargs": """kwargs_list : Sized[dict[str, Any]] | None
    The keyword arguments for the function. Each task should have its own
    dictionary of arguments or give None to use an empty dictionary for
    all.""",
    "n_jobs": """n_jobs : int | None
    Number of CPU cores to be used. If None, use all physical cores. If -1,
    use all logical cores. If 0 or 1, run in serial. If negative, use all
    logical cores + 1 + n_jobs.""",
    "verbose": """verbose : int, optional
    Verbosity level for joblib backend (default is 0).""",
    "backend": """backend : str, optional
    The backend to use for parallel execution. Allowed values are
    'for_loop', 'list_comprehension', 'async_job', 'threading_job', 'poo',
    'process', 'thread_pool_executor', 'process_pool_executor',
    'joblib' (='joblib_loky'), 'joblib_threading' and 'subinterpreter'
    (default is "for_loop").""",
    "i": """i : int
    The index of the task.""",
    "measure_time": """measure_time : bool
    Whether to measure time for each task. If True, each job result has a
    tuple of (original_result, start_end_tuple). (default is False).""",
}

doc_parallel = f"""
Parameters
----------
{docs['func']}
{docs['args_list']}
{docs['kwargs_list']}
{docs['n_jobs']}
{docs['verbose']}
{docs['measure_time']}

Returns
-------
tuple
    A tuple of results.
"""

doc_parallel_wrapper = f"""
Parameters
----------
{docs['func']}
{docs['args_list']}
{docs['kwargs_list']}
{docs['n_jobs']}
{docs['verbose']}
{docs['measure_time']}
{docs['backend']}

Returns
-------
tuple[Any] | tuple[tuple[Any], tuple[datetime, datetime]]
    A tuple of results. If measure_time is True, a tuple of results and start and end times is also returned.
"""
