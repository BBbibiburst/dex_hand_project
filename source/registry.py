# -*- coding: utf-8 -*-
"""Small reusable typed registry used by robots and task plug-ins."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(
        self,
        kind: str,
        *,
        normalize: Callable[[str], str] | None = None,
    ) -> None:
        self.kind = kind
        self._normalize = normalize or (lambda value: value)
        self._items: dict[str, T] = {}

    def register(self, name: str, value: T, *, replace: bool = False) -> T:
        key = self._key(name)
        previous = self._items.get(key)
        if previous is not None and previous is not value and not replace:
            raise KeyError(f"{self.kind.title()} {key!r} is already registered.")
        self._items[key] = value
        return value

    def get(self, name: str) -> T:
        key = self._key(name)
        try:
            return self._items[key]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} {name!r}. Available {self.kind}s: {available}."
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def values(self) -> Iterable[T]:
        return self._items.values()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._key(name) in self._items

    def _key(self, name: str) -> str:
        key = self._normalize(name.strip())
        if not key:
            raise ValueError(f"{self.kind.title()} name cannot be empty.")
        return key
