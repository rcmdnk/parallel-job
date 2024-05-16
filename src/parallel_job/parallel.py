from __future__ import annotations

import inspect
import textwrap
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING

from .doc import append_docstring, doc_parallel, doc_parallel_wrapper
from .utils import cpu_count

if TYPE_CHECKING:
    import multiprocessing.queues
    import multiprocessing.synchronize
    import queue
    import threading
    from typing import Any, Callable

    ArgsList = (
        list[tuple[Any, ...]] | tuple[tuple[Any, ...]] | set[tuple[Any, ...]]
    )
    KwargsList = (
        list[dict[str, Any]] | tuple[dict[str, Any]] | set[dict[str, Any]]
    )


def measure_func_time(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, tuple[datetime, datetime]]:
    """
    Wrapper to execute a function and capture its start and end times.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to execute.
    args : tuple[Any]
        The positional arguments for the function.
    kwargs : dict[str, Any]
        The keyword arguments for the function.

    Returns
    -------
    tuple
        A tuple containing the function's return value and its start and end times.
    """
    start = datetime.now()
    ret = func(*args, **kwargs)
    end = datetime.now()
    return ret, (start, end)


def fix_parallel_args(func_type: str = "Callable") -> Callable[..., Any]:
    def _deco(parallel_func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(
            func: Callable[..., Any],
            args_list: ArgsList,
            kwargs_list: KwargsList | None = None,
            n_jobs: int = -1,
            verbose: int = 0,
            measure_time: bool = False,
        ) -> tuple[Any]:
            if kwargs_list is None:
                kwargs_list = [{}] * len(args_list)
            if len(kwargs_list) != len(args_list):
                raise ValueError(
                    f"args and kwargs_list must have the same length ({len(args_list)} != {len(kwargs_list)})."
                )
            n_jobs = cpu_count(n_jobs)
            _func: Callable[..., Any] | str = func
            if measure_time:
                if func_type == "Callable":
                    _func = partial(measure_func_time, func=func)
                elif func_type == "str":
                    _func = f"""
from datetime import datetime

{textwrap.dedent(inspect.getsource(func))}
start = datetime.now()
ret = {func.__name__}(*args, **kwargs)
end = datetime.now()
result = ret, (start, end)
"""
            elif func_type == "str":
                _func = f"""
{textwrap.dedent(inspect.getsource(func))}
result = {func.__name__}(*args, **kwargs)
"""
            return parallel_func(
                _func, args_list, kwargs_list, n_jobs, verbose
            )

        wrapper.__doc__ = parallel_func.__doc__
        return wrapper

    return _deco


def queue_to_results(
    output_queue: queue.Queue[Any] | multiprocessing.queues.Queue[Any],
) -> tuple[Any]:
    results = {}
    while not output_queue.empty():
        index, ret = output_queue.get()
        results[index] = ret
    return tuple(v for k, v in sorted(results.items()))


def run_thread(
    target: Callable[..., Any],
    func: Callable[..., Any] | str,
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
) -> tuple[Any]:
    """
    Run tasks in multiple threads and collect their results.

    Parameters
    ----------
    target : Callable[..., Any]
        The target function for threading.
    func : Callable[..., Any] | str
        The function to execute.
    args_list : list of tuple
        List of positional arguments for each task.
    kwargs_list : list of dict
        List of keyword arguments for each task.
    n_jobs : int
        Number of concurrent threads.

    Returns
    -------
    tuple
        A tuple containing a tuple of results and a tuple of start and end times.
    """
    import queue
    import threading

    sem = threading.Semaphore(n_jobs)
    output_queue: queue.Queue[Any] = queue.Queue()
    threads = [
        threading.Thread(
            target=target, args=(func, args, kwargs, i, output_queue, sem)
        )
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return queue_to_results(output_queue)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def for_loop(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in a for loop and collect their results.
    """
    results = []
    for args, kwargs in zip(args_list, kwargs_list):
        results.append(func(*args, **kwargs))
    return tuple(results)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def list_comprehension(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks using list comprehension and collect their results.
    """
    return tuple(
        func(*args, **kwargs) for args, kwargs in zip(args_list, kwargs_list)
    )


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def async_job(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks asynchronously and collect their results.
    """
    import asyncio

    sem = asyncio.Semaphore(n_jobs)

    async def async_func(
        args: tuple[Any, ...], kwargs: dict[str, Any], sem: asyncio.Semaphore
    ) -> Any:
        loop = asyncio.get_event_loop()
        async with sem:
            return await loop.run_in_executor(
                None,
                partial(func, *args, **kwargs),
            )

    async def async_main() -> list[Any]:
        tasks = [
            async_func(args, kwargs, sem)
            for args, kwargs in zip(args_list, kwargs_list)
        ]
        return await asyncio.gather(*tasks)

    return tuple(asyncio.run(async_main()))


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def threading_job(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in multiple threads and collect their results.
    """

    def target(
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        i: int,
        output_queue: queue.Queue[Any] | multiprocessing.queues.Queue[Any],
        sem: threading.Semaphore,
    ) -> None:
        with sem:
            ret = func(*args, **kwargs)
            output_queue.put((i, ret))

    return run_thread(target, func, args_list, kwargs_list, n_jobs)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def pool(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in a pool of workers and collect their results.
    """
    import multiprocessing

    with multiprocessing.Pool(processes=n_jobs) as pool:
        ret = pool.starmap(
            func,
            [(args, kwargs) for args, kwargs in zip(args_list, kwargs_list)],
        )
    return tuple(ret)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def process(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in multiple processes and collect their results.
    """
    import multiprocessing

    output_queue: multiprocessing.queues.Queue[Any] = multiprocessing.Queue()
    sem = multiprocessing.Semaphore(n_jobs)

    def target(
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        i: int,
        output_queue: multiprocessing.queues.Queue[Any],
        sem: multiprocessing.synchronize.Semaphore,
    ) -> None:
        with sem:
            output_queue.put(i, func(*args, **kwargs))

    processes = [
        multiprocessing.Process(
            target=target,
            args=(func, args, kwargs, i, output_queue, sem),
        )
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    return queue_to_results(output_queue)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def thread_pool_executor(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in a thread pool executor and collect their results.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        ret = [
            executor.submit(func, args, kwargs).result()
            for args, kwargs in zip(args_list, kwargs_list)
        ]
    return tuple(ret)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def process_pool_executor(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in a process pool executor and collect their results.
    """
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        ret = [
            executor.submit(func, args, kwargs).result()
            for args, kwargs in zip(args_list, kwargs_list)
        ]
    return tuple(ret)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def joblib_loky(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in joblib using the loky backend and collect their results.
    """
    import joblib

    ret = joblib.Parallel(n_jobs, backend="loky", verbose=verbose)(
        joblib.delayed(func)(args, kwargs)
        for args, kwargs in zip(args_list, kwargs_list)
    )
    return tuple(ret)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="Callable")
def joblib_threading(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in joblib using the threading backend and collect their results.
    """
    import joblib

    ret = joblib.Parallel(n_jobs, backend="threading", verbose=verbose)(
        joblib.delayed(func)(args, kwargs)
        for args, kwargs in zip(args_list, kwargs_list)
    )
    return tuple(ret)


@append_docstring(doc_parallel)
@fix_parallel_args(func_type="str")
def subinterpreter(
    func: str,
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int,
    verbose: int,
) -> tuple[Any]:
    """
    Run tasks in subinterpreters and collect their results.
    """
    import ctypes
    import json

    import _xxsubinterpreters as subinterpreters

    class SharedResult(ctypes.Structure):
        fields = [("data", ctypes.c_char_p)]

    shared_result = SharedResult()

    def target(
        code: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        i: int,
        output_queue: queue.Queue[Any],
        sem: threading.Semaphore,
    ) -> None:
        code = f"""
import json
import ctypes

args = {json.dumps(args)}
kwargs = {json.dumps(kwargs)}
{code}
result_json = json.dumps(result)
shared_result = ctypes.cast({ctypes.addressof(shared_result)}, ctypes.POINTER(ctypes.c_char_p))
shared_result.contents.value = result_json.encode('utf-8')
"""

        with sem:
            interp_id = subinterpreters.create()
            subinterpreters.run_string(interp_id, code)
            subinterpreters.destroy(interp_id)
            result_json = shared_result.data.decode("utf-8")
            output_queue.put(i, json.loads(result_json))

    return run_thread(target, func, args_list, kwargs_list, n_jobs)


@append_docstring(doc_parallel_wrapper)
def parallel_wrapper(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList | None,
    n_jobs: int = -1,
    verbose: int = 0,
    measure_time: bool = False,
    backend: str = "for_loop",
) -> tuple[tuple[Any], tuple[datetime, datetime]] | tuple[Any]:
    """
    Wrapper to execute a function in parallel using different backends.
    """
    match backend:
        case "for_loop":
            parallel_func = for_loop
        case "list_comprehension":
            parallel_func = list_comprehension
        case "async_job":
            parallel_func = async_job
        case "threading_job":
            parallel_func = threading_job
        case "pool":
            parallel_func = pool
        case "process":
            parallel_func = process
        case "thread_pool_executor":
            parallel_func = thread_pool_executor
        case "process_pool_executor":
            parallel_func = process_pool_executor
        case _ if "joblib" in backend:
            if "threading" in backend:
                parallel_func = joblib_threading
            else:
                parallel_func = joblib_loky
        case "subinterpreter":
            parallel_func = subinterpreter
        case _:
            raise ValueError(f"backend {backend} is not supported")
    if measure_time:
        from datetime import datetime

        start = datetime.now()
    results = parallel_func(
        func, args_list, kwargs_list, n_jobs, verbose, measure_time
    )
    if measure_time:
        end = datetime.now()
        return results, (start, end)
    return results
