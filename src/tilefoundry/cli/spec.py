"""The `spec` command: which specifications exist, what is in one, and one section."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tilefoundry.cli import data
from tilefoundry.utils.markdown import headings

#: Topic names that differ from their document's filename.
_SPEC_TOPICS = {
    "cli": "cli",
    "dsl": "hir",
}

#: The document title is not a section anybody asks for by name.
_ADDRESSABLE = 2


def spec_path(topic: str) -> Path:
    """The document a topic names, from this checkout or from the installation."""
    return data.path("spec", f"{_SPEC_TOPICS.get(topic, topic)}.md")


def read_spec(topic: str) -> str:
    """Read the document a topic names."""
    return spec_path(topic).read_text(encoding="utf-8")


def spec_directory() -> Path:
    """The directory the documents are read from, wherever they were found."""
    return data.directory("spec")


def topics() -> dict[str, Path]:
    """Every document that can be asked for, by the name it is asked for under."""
    found = {path.stem: path for path in sorted(spec_directory().glob("*.md"))}
    for topic, name in _SPEC_TOPICS.items():
        if name in found:
            found.setdefault(topic, found[name])
    return found


@dataclass(frozen=True)
class Section:
    """One heading and the lines under it, down to the next heading as high."""

    key: str
    title: str
    level: int
    start: int
    end: int


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.replace("`", "").lower()).strip("-")


def _key_and_title(heading) -> tuple[str, str]:
    if heading.number is not None:
        return heading.number, heading.title
    # An unnumbered heading still has to be addressable: most of the op catalogue
    # is unnumbered, and a section nobody can name is a section nobody can read.
    return _slug(heading.title), heading.title


def _disambiguate(bases: list[str], ancestries: list[tuple[str, ...]]) -> list[str]:
    """One key per section, no two alike.

    A clash takes on the name of its enclosing heading, then the one above that,
    until the keys separate.
    """
    from collections import Counter  # noqa: PLC0415

    def compose(index: int, depth: int) -> str:
        ancestry = ancestries[index]
        return "/".join((*ancestry[len(ancestry) - depth :], bases[index]))

    depths = [0] * len(bases)
    keys = [compose(index, 0) for index in range(len(bases))]
    while True:
        counts = Counter(keys)
        clashing = [index for index, key in enumerate(keys) if counts[key] > 1]
        deepened = [
            index for index in clashing if depths[index] < len(ancestries[index])
        ]
        if not deepened:
            # Two headings with the same name and the same ancestry are only
            # distinguishable by which came first.
            seen: Counter[str] = Counter()
            for index in clashing:
                seen[keys[index]] += 1
                if seen[keys[index]] > 1:
                    keys[index] = f"{keys[index]}#{seen[keys[index]]}"
            return keys
        for index in deepened:
            depths[index] += 1
            keys[index] = compose(index, depths[index])


def sections(text: str) -> tuple[Section, ...]:
    """Every addressable section of *text*, in document order."""
    lines = text.splitlines()
    scanned: list[tuple[int, str, str, int]] = []
    ancestries: list[tuple[str, ...]] = []
    enclosing: dict[int, str] = {}
    for heading in headings(text):
        if heading.level < _ADDRESSABLE:
            continue
        level = heading.level
        key, title = _key_and_title(heading)
        enclosing = {at: slug for at, slug in enclosing.items() if at < level}
        ancestries.append(tuple(enclosing[at] for at in sorted(enclosing)))
        enclosing[level] = _slug(title)
        scanned.append((level, key, title, heading.line))

    keys = _disambiguate([key for _, key, _, _ in scanned], ancestries)
    found = [
        Section(key=key, title=title, level=level, start=start, end=len(lines))
        for key, (level, _, title, start) in zip(keys, scanned)
    ]
    # A section ends where the next one at its level or above begins.
    for index, section in enumerate(found):
        for later in found[index + 1 :]:
            if later.level <= section.level:
                found[index] = Section(
                    key=section.key, title=section.title, level=section.level,
                    start=section.start, end=later.start,
                )
                break
    return tuple(found)


def render_topics() -> str:
    """The documents there are, and which name reaches each."""
    found = topics()
    aliases = {
        topic: name for topic, name in _SPEC_TOPICS.items() if topic != name
    }
    width = max(len(name) for name in found)
    lines = [f"Specifications in {spec_directory()}:", ""]
    for name in sorted(found):
        alias = aliases.get(name)
        row = f"  {name:<{width}}  another name for {alias}" if alias else f"  {name}"
        lines.append(row)
    lines += ["", "Ask for one with `tilefoundry spec <topic>`, a section with"]
    lines.append("`tilefoundry spec <topic> <section>`.")
    return "\n".join(lines) + "\n"


def render_outline(topic: str) -> str:
    """What is in one document: every section's key and title, indented by level."""
    found = sections(read_spec(topic))
    if not found:
        return f"{topic}: no sections\n"
    # Capped: one long key would otherwise push every title across the screen.
    width = min(24, max(len(section.key) for section in found))
    lines = [f"{topic} ({spec_path(topic).name}):", ""]
    for section in found:
        indent = "  " * (section.level - 1)
        lines.append(f"  {section.key:<{width}}  {indent}{section.title}")
    return "\n".join(lines) + "\n"


def _neighbours(found: tuple[Section, ...], index: int) -> list[str]:
    """The sections beside this one, so a reader can walk without the outline."""
    chosen = found[index]
    siblings = [
        section for section in found if section.level == chosen.level
    ]
    at = siblings.index(chosen)
    beside = []
    if at:
        beside.append(f"previous: {siblings[at - 1].key}  {siblings[at - 1].title}")
    if at + 1 < len(siblings):
        beside.append(f"next:     {siblings[at + 1].key}  {siblings[at + 1].title}")
    inside = [
        section for section in found[index + 1 :]
        if section.level == chosen.level + 1 and section.start < chosen.end
    ]
    if inside:
        beside.append("inside:   " + ", ".join(section.key for section in inside))
    return beside


def render_section(topic: str, key: str) -> str:
    """One section of one document, and the keys of the sections beside it."""
    text = read_spec(topic)
    found = sections(text)
    for index, section in enumerate(found):
        if section.key == key:
            body = "\n".join(text.splitlines()[section.start : section.end]).rstrip()
            beside = _neighbours(found, index)
            if not beside:
                return body + "\n"
            return body + "\n\n" + "\n".join(f"# {line}" for line in beside) + "\n"
    keys = ", ".join(section.key for section in found)
    raise ValueError(f"{topic} has no section {key!r}; it has {keys}")


def run_spec(topic: str | None, section: str | None) -> int:
    """Print the documents, one document's outline, or one of its sections."""
    import sys  # noqa: PLC0415

    if topic is None:
        sys.stdout.write(render_topics())
    elif section is None:
        sys.stdout.write(render_outline(topic))
    else:
        sys.stdout.write(render_section(topic, section))
    return 0


__all__ = [
    "Section",
    "read_spec",
    "render_outline",
    "render_section",
    "render_topics",
    "run_spec",
    "sections",
    "spec_path",
    "topics",
]
