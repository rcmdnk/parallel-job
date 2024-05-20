from typing import Any

ArgsList = (
    list[tuple[Any, ...]] | tuple[tuple[Any, ...]] | set[tuple[Any, ...]]
)
KwargsList = list[dict[str, Any]] | tuple[dict[str, Any]] | set[dict[str, Any]]
