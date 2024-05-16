def num_string(x: int | float) -> str:
    """
    Format a number as a string, using 0 decimal places if >= 1000, otherwise 3 significant figures.

    Parameters
    ----------
    x : int or float
        The number to format.

    Returns
    -------
    str
        The formatted number.
    """
    return f"{x:.0f}" if x >= 1000 else f"{x:.3g}"


def cpu_count(n_jobs: int | None) -> int:
    """
    Return number of CPU cores to be used based on the input.

    Parameters
    ----------
    n_jobs : int or None
        Number of CPU cores to be used. If None, use all physical cores. If -1, use all logical cores.
        If 0 or 1, run in serial. If negative, use all logical cores + 1 + n_jobs.

    Returns
    -------
    int
        Adjusted number of CPU cores.
    """
    if n_jobs is None:
        import joblib

        return joblib.cpu_count(only_physical_cores=True)
    if n_jobs < 0:
        import os

        n_cpu = os.cpu_count() or 1
        return n_cpu + 1 + n_jobs
    return max(n_jobs, 1)
