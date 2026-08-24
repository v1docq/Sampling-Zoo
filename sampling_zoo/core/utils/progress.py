from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    enabled: bool = True,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    leave: bool = False,
    **kwargs: Any,
) -> Iterator[T]:
    """Wrap an iterable with tqdm while keeping call sites quiet when disabled."""

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=not enabled,
        leave=leave,
        **kwargs,
    )


@contextmanager
def progress_bar(
    *,
    enabled: bool = True,
    desc: Optional[str] = None,
    total: int = 1,
    leave: bool = False,
    **kwargs: Any,
):
    bar = tqdm(
        total=total,
        desc=desc,
        disable=not enabled,
        leave=leave,
        **kwargs,
    )
    try:
        yield bar
    finally:
        bar.close()


def progress_write(message: str, *, enabled: bool = True) -> None:
    if enabled:
        tqdm.write(message)
    else:
        print(message)
