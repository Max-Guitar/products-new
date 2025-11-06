"""Minimal BeautifulSoup compatibility shim.

This lightweight implementation provides just enough of the
``BeautifulSoup`` API for the sanitisation helpers in the Streamlit app.
It is **not** a drop-in replacement for the full library, but mirrors the
behaviour the app relies on: parsing HTML snippets, iterating over tags,
unwrapping disallowed nodes and rendering the DOM back to a string with
closed tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from html import escape
from typing import Iterable, Iterator, Optional, Union


Node = Union["Tag", str]


@dataclass
class Tag:
    name: Optional[str]
    attrs: dict[str, str] = field(default_factory=dict)
    parent: Optional["Tag"] = None
    children: list[Node] = field(default_factory=list)

    def append(self, child: Node) -> None:
        self.children.append(child)
        if isinstance(child, Tag):
            child.parent = self

    def find_all(self, name: Optional[str] = None) -> list["Tag"]:
        results: list[Tag] = []
        for child in self.children:
            if isinstance(child, Tag):
                if name in (None, True) or child.name == name:
                    results.append(child)
                results.extend(child.find_all(name))
        return results

    def unwrap(self) -> None:
        if not self.parent:
            return
        siblings = self.parent.children
        try:
            index = siblings.index(self)
        except ValueError:
            return
        for child in self.children:
            if isinstance(child, Tag):
                child.parent = self.parent
        siblings[index:index + 1] = self.children
        self.parent = None
        self.children = []

    def _serialize(self) -> str:
        if self.name is None:
            return "".join(
                child if isinstance(child, str) else child._serialize()
                for child in self.children
            )
        attr_bits: list[str] = []
        for key, value in self.attrs.items():
            if value is None:
                attr_bits.append(f" {key}")
            else:
                attr_bits.append(f" {key}=\"{escape(str(value), quote=True)}\"")
        inner = "".join(
            child if isinstance(child, str) else child._serialize()
            for child in self.children
        )
        return f"<{self.name}{''.join(attr_bits)}>{inner}</{self.name}>"

    def __str__(self) -> str:  # pragma: no cover - delegated to _serialize
        return self._serialize()


class _SoupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.root = Tag(name=None)
        self.current = self.root

    # HTMLParser interface -------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Tag(name=tag, attrs={k: v for k, v in attrs if k})
        self.current.append(node)
        self.current = node

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        node = Tag(name=tag, attrs={k: v for k, v in attrs if k})
        self.current.append(node)

    def handle_endtag(self, tag: str) -> None:
        cursor = self.current
        while cursor is not None and cursor.name != tag and cursor.parent is not None:
            cursor = cursor.parent
        if cursor is None:
            return
        self.current = cursor.parent or self.root

    def handle_data(self, data: str) -> None:
        if data:
            self.current.append(data)

    def handle_entityref(self, name: str) -> None:
        self.current.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.current.append(f"&#{name};")

    def close(self) -> None:
        super().close()
        while self.current.parent is not None:
            self.current = self.current.parent


class BeautifulSoup:
    def __init__(self, markup: str, parser: str = "html.parser") -> None:
        if parser != "html.parser":  # pragma: no cover - defensive guard
            raise ValueError("Only the built-in html.parser is supported")
        self._parser = _SoupParser()
        self._parser.feed(markup or "")
        self._parser.close()
        self.root = self._parser.root

    def find_all(self, name: Optional[str] = None) -> list[Tag]:
        return self.root.find_all(name)

    def __iter__(self) -> Iterator[Tag]:  # pragma: no cover - compatibility
        return iter(self.root.children)  # type: ignore[return-value]

    def __str__(self) -> str:  # pragma: no cover - delegated to Tag
        return str(self.root)

