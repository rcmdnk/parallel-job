from __future__ import annotations

import inspect
import textwrap
from functools import partial, wraps
from typing import TYPE_CHECKING

from _pydatetime import datetime

from .doc import (
    append_docstring,
    doc_joblib_params,
    doc_parallel,
    doc_parallel_wrapper,
    doc_subinterpreter_params,
)
from .type_helper import ArgsList, KwargsList
from .utils import cpu_count

if TYPE_CHECKING:
    import multiprocessing.queues
    import multiprocessing.synchronize
    import queue
    import threading
    from typing import Any, Callable


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


def arg_wrapper(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """
    Wrapper to execute a function with its arguments.

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
    Any
        The function's return value.
    """
    return func(*args, **kwargs)


def fix_parallel_args(func_type: str = "Callable") -> Callable[..., Any]:
    def _deco(parallel_func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(parallel_func)
        parameters = list(sig.parameters.values())
        new_param = inspect.Parameter(
            "measure_time",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=False,
            annotation=bool,
        )
        parameters.append(new_param)

        index = next(
            i for i, p in enumerate(parameters) if p.name == "kwargs_list"
        )
        kwargs_list = parameters[index]

        new_kwargs_list = kwargs_list.replace(
            annotation=KwargsList | None, default=None
        )
        parameters[index] = new_kwargs_list

        new_sig = sig.replace(parameters=parameters)

        @wraps(parallel_func)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any]:
            bound_args = new_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            measure_time = bound_args.arguments.pop("measure_time")

            args_list = bound_args.arguments["args_list"]
            kwargs_list = bound_args.arguments.get("kwargs_list")
            if kwargs_list is None:
                kwargs_list = [{}] * len(args_list)
            if len(kwargs_list) != len(args_list):
                raise ValueError(
                    f"args_list and kwargs_list must have the same length ({len(args_list)} != {len(kwargs_list)})."
                )
            bound_args.arguments["kwargs_list"] = kwargs_list

            bound_args.arguments["n_jobs"] = cpu_count(
                bound_args.arguments["n_jobs"]
            )
            func = bound_args.arguments["func"]
            _func: Callable[..., Any] | str = func
            if measure_time:
                if func_type == "Callable":
                    _func = partial(measure_func_time, func)
                elif func_type == "str":
                    _func = f"""
from _pydatetime import datetime

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
            bound_args.arguments["func"] = _func
            return parallel_func(*bound_args.args, **bound_args.kwargs)

        wrapper.__signature__ = new_sig  # type: ignore
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
            target=target,
            kwargs={
                "func": func,
                "args": args,
                "kwargs": kwargs,
                "i": i,
                "output_queue": output_queue,
                "sem": sem,
            },
        )
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return queue_to_results(output_queue)


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def for_loop(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[Any]:
    """
    Run tasks in a for loop and collect their results.
    """
    results = []
    for args, kwargs in zip(args_list, kwargs_list):
        results.append(func(*args, **kwargs))
    return tuple(results)


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def list_comprehension(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[Any]:
    """
    Run tasks using list comprehension and collect their results.
    """
    return tuple(
        func(*args, **kwargs) for args, kwargs in zip(args_list, kwargs_list)
    )


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def async_run(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
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


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def threading_run(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
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
            output_queue.put((i, func(*args, **kwargs)))

    return run_thread(target, func, args_list, kwargs_list, n_jobs)


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def pool(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[Any]:
    """
    Run tasks in a pool of workers and collect their results.
    """
    import multiprocessing

    with multiprocessing.Pool(processes=n_jobs) as pool:
        ret = pool.starmap(
            arg_wrapper,
            [
                (func, args, kwargs)
                for args, kwargs in zip(args_list, kwargs_list)
            ],
        )
    return tuple(ret)


def process_target(
    func: Callable[..., Any],
    i: int,
    sem: multiprocessing.synchronize.Semaphore,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output_queue: multiprocessing.queues.Queue[Any],
) -> None:
    with sem:
        output_queue.put((i, func(*args, **kwargs)))


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def process(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[Any]:
    """
    Run tasks in multiple processes and collect their results.
    """
    import multiprocessing

    output_queue: multiprocessing.queues.Queue[Any] = multiprocessing.Queue()
    sem = multiprocessing.Semaphore(n_jobs)

    processes = [
        multiprocessing.Process(
            target=process_target,
            args=(func, i, sem, args, kwargs, output_queue),
        )
        for i, (args, kwargs) in enumerate(zip(args_list, kwargs_list))
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    return queue_to_results(output_queue)


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def thread_pool_executor(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[Any]:
    """
    Run tasks in a thread pool executor and collect their results.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        features = [
            executor.submit(func, *args, **kwargs)
            for args, kwargs in zip(args_list, kwargs_list)
        ]
        ret = tuple(feature.result() for feature in features)
    return ret


@append_docstring(doc_parallel.format(additional_params=""))
@fix_parallel_args(func_type="Callable")
def process_pool_executor(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple[Any]:
    """
    Run tasks in a process pool executor and collect their results.
    """
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        features = [
            executor.submit(func, *args, **kwargs)
            for args, kwargs in zip(args_list, kwargs_list)
        ]
        ret = tuple(feature.result() for feature in features)
    return ret


@append_docstring(doc_parallel.format(additional_params=doc_joblib_params))
@fix_parallel_args(func_type="Callable")
def joblib_run(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
    backend: str = "loky",
) -> tuple[Any]:
    """
    Run tasks in joblib and collect their results.
    """
    import joblib

    ret = joblib.Parallel(n_jobs, backend=backend, verbose=verbose)(
        joblib.delayed(func)(*args, **kwargs)
        for args, kwargs in zip(args_list, kwargs_list)
    )
    return tuple(ret)


def subinterpreter_target(
    func: str,
    i: int,
    sem: threading.Semaphore,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output_queue: queue.Queue[Any],
    arg_manager: str,
    result_manager: str,
) -> None:
    import _xxsubinterpreters as subinterpreters

    with sem:
        interp_id = subinterpreters.create()
        shared = {}
        match arg_manager:
            case "pickle":
                import pickle

                func = f"""
import pickle

args = pickle.loads({pickle.dumps(args)!r})
kwargs = pickle.loads({pickle.dumps(kwargs)!r})
{func}
"""
            case "json":
                import json

                func = f"""
args = {json.dumps(args)}
kwargs = {json.dumps(kwargs)}
{func}
"""
            case "shared":
                if kwargs:
                    raise ValueError(
                        "shared argument manager is not supported with kwargs"
                    )
                shared = {"args": args}
                func = f"""
kwargs = {{}}
{func}
"""  # noqa: P103
            case _:
                raise ValueError(f"arg_manager {arg_manager} is not supported")
        match result_manager:
            case "":
                pass
            case "asis":
                import _xxinterpchannels as channels

                channel_id = channels.create()
                func = f"""{func}
import _xxinterpchannels as channels
channels.send({channel_id}, result)
"""
            case "pickle":
                import _xxinterpchannels as channels

                channel_id = channels.create()
                func = f"""{func}
import _xxinterpchannels as channels
import pickle
channels.send({channel_id}, pickle.dumps(result))
"""
            case "json":
                import _xxinterpchannels as channels

                channel_id = channels.create()
                func = f"""{func}
import _xxinterpchannels as channels
import json
channels.send({channel_id}, json.dumps(result))
"""

        subinterpreters.run_string(interp_id, func, shared=shared)
        match result_manager:
            case "":
                result = (None,)
            case "asis":
                result = channels.recv(channel_id)
            case "pickle":
                import pickle

                result = pickle.loads(channels.recv(channel_id))
            case "json":
                import json

                result = json.loads(channels.recv(channel_id))
        output_queue.put((i, result))


@append_docstring(
    doc_parallel.format(additional_params=doc_subinterpreter_params)
)
@fix_parallel_args(func_type="str")
def subinterpreter(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList,
    n_jobs: int = -1,
    verbose: int = 0,
    arg_manager: str = "json",
    result_manager: str = "json",
) -> tuple[Any]:
    """
    Run tasks in subinterpreters and collect their results.
    """
    target = partial(
        subinterpreter_target,
        arg_manager=arg_manager,
        result_manager=result_manager,
    )
    return run_thread(target, func, args_list, kwargs_list, n_jobs)


@append_docstring(doc_parallel_wrapper)
def parallel_wrapper(
    func: Callable[..., Any],
    args_list: ArgsList,
    kwargs_list: KwargsList | None,
    n_jobs: int = -1,
    verbose: int = 0,
    backend: str = "for_loop",
    measure_time: bool = False,
) -> tuple[tuple[Any], tuple[datetime, datetime]] | tuple[Any]:
    """
    Wrapper to execute a function in parallel using different backends.
    """
    match backend:
        case "for_loop":
            parallel_func = for_loop
        case "list_comprehension":
            parallel_func = list_comprehension
        case "async_run":
            parallel_func = async_run
        case "threading_run":
            parallel_func = threading_run
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
                parallel_func = partial(joblib_run, backend="threading")
            else:
                parallel_func = partial(joblib_run, backend="loky")
        case _ if "subinterpreter" in backend:
            params = backend.split("_")
            if len(params) == 1:
                parallel_func = subinterpreter
            elif len(params) == 2:
                parallel_func = partial(subinterpreter, arg_manager=params[1])
            else:
                parallel_func = partial(
                    subinterpreter,
                    arg_manager=params[1],
                    result_manager=params[2],
                )
        case _:
            raise ValueError(f"backend {backend} is not supported")
    if measure_time:
        start = datetime.now()
    results = parallel_func(
        func=func,
        args_list=args_list,
        kwargs_list=kwargs_list,
        n_jobs=n_jobs,
        verbose=verbose,
        measure_time=measure_time,
    )
    if measure_time:
        end = datetime.now()
        return results, (start, end)
    return results
