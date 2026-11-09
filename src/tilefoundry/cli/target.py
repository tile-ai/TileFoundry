"""Persist, list, and inspect Target providers and hardware documents."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from tilefoundry import target, utils
from tilefoundry.target import hardware


class KnownModuleTargets:
    """The Target classes each provider module registered, by module name."""

    def __init__(self) -> None:
        self.by_module: dict[str, tuple[type[target.Target], ...]] = {}

    def remember(self, module_name: str, target_types: tuple[type[target.Target], ...]) -> None:
        """Record the Target classes *module_name* registered."""
        self.by_module[module_name] = target_types

    def recall(self, module_name: str) -> tuple[type[target.Target], ...]:
        """The Target classes *module_name* registered earlier, or none."""
        return self.by_module.get(module_name, ())


known_module_targets = KnownModuleTargets()


@dataclass(frozen=True)
class DocumentEntry:
    id: str
    source: Path
    digest: str


@dataclass(frozen=True)
class ModuleEntry:
    name: str | None = None
    source: Path | None = None

    @property
    def key(self) -> str:
        if self.name is not None:
            return self.name
        if self.source is None:
            raise ValueError("module entry has neither a name nor a source")
        return self.source.stem


@dataclass(frozen=True)
class RegistryState:
    documents: tuple[DocumentEntry, ...] = ()
    modules: tuple[ModuleEntry, ...] = ()


@dataclass(frozen=True)
class LoadedDocument:
    entry: DocumentEntry
    owner: type[target.Target]
    document: hardware.Document


@dataclass(frozen=True)
class LoadedModule:
    entry: ModuleEntry
    target_types: tuple[type[target.Target], ...]
    identities: tuple[str, ...]


@dataclass(frozen=True)
class LoadedRegistrations:
    """Persistent entries together with the ones successfully replayed."""

    path: Path
    state: RegistryState
    documents: tuple[LoadedDocument, ...] = ()
    modules: tuple[LoadedModule, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def added_identities(self) -> frozenset[str]:
        document_ids = {
            loaded.document.id
            for loaded in self.documents
            if loaded.document.kind == "device"
        }
        module_ids = {
            identity for loaded in self.modules for identity in loaded.identities
        }
        return frozenset((*document_ids, *module_ids))


def registry_path(override: str | Path | None = None) -> Path:
    """The writable registry belonging to this Python installation."""
    if override is not None:
        return Path(override).expanduser().resolve()
    return Path(sys.prefix) / "share" / "tilefoundry" / "registry.toml"


def string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"registry {field} must be a non-empty string")
    return value


def read_registry(path: Path) -> RegistryState:
    if not path.exists():
        return RegistryState()
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path}: registry is not readable TOML: {error}") from error
    unknown = set(raw) - {"document", "module"}
    if unknown:
        raise ValueError(f"{path}: unknown registry tables {sorted(unknown)}")

    documents: list[DocumentEntry] = []
    for index, item in enumerate(raw.get("document", [])):
        if not isinstance(item, dict) or set(item) != {"id", "source", "digest"}:
            raise ValueError(
                f"{path}: document entry {index} must contain exactly id, source, digest"
            )
        documents.append(
            DocumentEntry(
                string(item["id"], field="document.id"),
                Path(string(item["source"], field="document.source")),
                string(item["digest"], field="document.digest"),
            )
        )

    modules: list[ModuleEntry] = []
    for index, item in enumerate(raw.get("module", [])):
        if not isinstance(item, dict) or set(item) not in ({"name"}, {"source"}):
            raise ValueError(
                f"{path}: module entry {index} must contain exactly name or source"
            )
        if "name" in item:
            modules.append(ModuleEntry(name=string(item["name"], field="module.name")))
        else:
            modules.append(
                ModuleEntry(source=Path(string(item["source"], field="module.source")))
            )
    return RegistryState(tuple(documents), tuple(modules))


def write_registry(path: Path, state: RegistryState) -> None:
    lines: list[str] = []
    for entry in state.documents:
        lines.extend(
            (
                "[[document]]",
                f"id = {json.dumps(entry.id)}",
                f"source = {json.dumps(str(entry.source))}",
                f"digest = {json.dumps(entry.digest)}",
                "",
            )
        )
    for entry in state.modules:
        lines.append("[[module]]")
        if entry.name is not None:
            lines.append(f"name = {json.dumps(entry.name)}")
        else:
            lines.append(f"source = {json.dumps(str(entry.source))}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def hardware_owner(schema: str) -> type[target.Target]:
    owners = tuple(
        target_type
        for target_type in target.registered().values()
        if "hardware" in vars(target_type)
        and isinstance(target_type.hardware, target.HardwareSpec)
        and target_type.hardware.supports_schema(schema)
    )
    if not owners:
        raise hardware.UnknownSchemaError(f"no Target owns hardware schema {schema!r}")
    if len(owners) != 1:
        names = sorted(f"{owner.__module__}.{owner.__qualname__}" for owner in owners)
        raise hardware.UnknownSchemaError(f"hardware schema {schema!r} has ambiguous owners {names}")
    return owners[0]


def module_occupant(name: str) -> Path | str | None:
    loaded = sys.modules.get(name)
    if loaded is not None:
        origin = getattr(loaded, "__file__", None)
        if origin is not None:
            return Path(origin).resolve()
        spec = getattr(loaded, "__spec__", None)
    else:
        spec = importlib.util.find_spec(name)
    if spec is None:
        return None if loaded is None else repr(loaded)
    if spec.origin not in (None, "built-in", "frozen"):
        return Path(spec.origin).resolve()
    if spec.origin is not None:
        return spec.origin
    locations = spec.submodule_search_locations
    if locations:
        return ", ".join(str(Path(location).resolve()) for location in locations)
    return repr(spec)


def load_module(entry: ModuleEntry) -> ModuleType:
    with contextlib.redirect_stdout(io.StringIO()):
        if entry.name is not None:
            return importlib.import_module(entry.name)
        path = entry.source
        if path is None or not path.is_file():
            raise FileNotFoundError(f"target module file {path} does not exist")
        name = path.stem
        occupant = module_occupant(name)
        if occupant is not None and occupant != path:
            raise ValueError(f"module name {name!r} is already taken by {occupant}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load target module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.update({name: module})
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module


def module_targets(
    module: ModuleType, *, replay_cached: bool
) -> tuple[type[target.Target], ...]:
    targets = tuple(
        target_type
        for target_type in target.registered().values()
        if target_type.__module__ == module.__name__
        or target_type.__module__.startswith(f"{module.__name__}.")
    )
    if not targets and replay_cached:
        targets = known_module_targets.recall(module.__name__)
        for target_type in targets:
            target.register(target_type)
    if not targets:
        raise ValueError(f"target module {module.__name__!r} registered no Target classes")
    return targets


def targets_of(target_types: tuple[type[target.Target], ...]) -> tuple[target.Target, ...]:
    return tuple(tgt for target_type in target_types for tgt in target_type.from_installed())


def identity_owners(exclude: frozenset[type[target.Target]] = frozenset()) -> dict[str, target.Target]:
    owners: dict[str, target.Target] = {}
    for tgt in available_targets():
        if type(tgt) in exclude:
            continue
        owners.setdefault(tgt.identity, tgt)
    return owners


def check_module_identities(
    target_types: tuple[type[target.Target], ...], existing: dict[str, target.Target]
) -> tuple[str, ...]:
    seen: dict[str, target.Target] = {}
    for tgt in targets_of(target_types):
        occupied = existing.get(tgt.identity) or seen.get(tgt.identity)
        if occupied is not None:
            raise ValueError(
                f"target identity {tgt.identity!r} is already occupied by "
                f"{occupied.to_python().text} ({type(occupied).__module__}.{type(occupied).__qualname__})"
            )
        seen[tgt.identity] = tgt
    return tuple(sorted(seen))


def rollback_targets(before: dict[str, type[target.Target]]) -> None:
    for name, target_type in tuple(target.registered().items()):
        if before.get(name) is not target_type:
            target.unregister(target_type)


def read_document(entry: DocumentEntry) -> hardware.Document:
    try:
        text = entry.source.read_text(encoding="utf-8")
    except OSError as error:
        raise OSError(f"document {entry.id!r} cannot read {entry.source}: {error}") from error
    document = hardware.parse_document(text, origin_label=str(entry.source))
    if document.id != entry.id:
        raise ValueError(
            f"document {entry.id!r} now declares ID {document.id!r}; add it again"
        )
    if document.digest != entry.digest:
        raise ValueError(
            f"document {entry.id!r} content changed at {entry.source}; add it again"
        )
    return document


def missing_architecture(owner: type[target.Target], document: hardware.Document) -> str | None:
    if document.kind != "device":
        return None
    available = owner.hardware.documents()
    return next(
        (
            item
            for item in document.compatibility
            if item not in available or available[item].kind != "architecture"
        ),
        None,
    )


def load_registrations(path: Path) -> LoadedRegistrations:
    """Replay each persistent entry, isolating one bad source from the others."""
    state = read_registry(path)
    loaded_documents: list[LoadedDocument] = []
    loaded_modules: list[LoadedModule] = []
    warnings: list[str] = []

    pending: list[tuple[DocumentEntry, hardware.Document]] = []
    for entry in state.documents:
        try:
            pending.append((entry, read_document(entry)))
        except Exception as error:
            warnings.append(str(error))
    pending.sort(key=lambda item: item[1].kind != "architecture")
    for entry, document in pending:
        try:
            owner = hardware_owner(document.schema)
            missing = missing_architecture(owner, document)
            if missing is not None:
                raise ValueError(
                    f"document {document.id!r} needs architecture {missing!r}; add it first"
                )
            existing = owner.hardware.documents().get(document.id)
            if existing is None:
                owner.hardware.adopt(document)
            elif existing.digest != document.digest:
                raise hardware.DuplicateRegistrationError(
                    f"hardware document {document.id!r} is already registered with different content"
                )
            loaded_documents.append(LoadedDocument(entry, owner, document))
        except Exception as error:
            warnings.append(str(error))

    for entry in state.modules:
        before = dict(target.registered())
        try:
            module = load_module(entry)
            target_types = module_targets(module, replay_cached=entry.name is not None)
            existing = identity_owners(frozenset(target_types))
            identities = check_module_identities(target_types, existing)
            known_module_targets.remember(module.__name__, target_types)
            loaded_modules.append(LoadedModule(entry, target_types, identities))
        except Exception as error:
            rollback_targets(before)
            warnings.append(f"module {entry.key!r} failed: {error}")
    return LoadedRegistrations(
        path,
        state,
        tuple(loaded_documents),
        tuple(loaded_modules),
        tuple(warnings),
    )


def available_targets() -> tuple[target.Target, ...]:
    """Every value currently constructible from registered Target classes."""
    return tuple(
        tgt
        for _, target_type in sorted(target.registered().items())
        for tgt in target_type.from_installed()
    )


def merged_imports(expressions: tuple[utils.PythonExpr, ...]) -> tuple[str, ...]:
    grouped: dict[str, set[str]] = {}
    other: set[str] = set()
    for statement in (item for expression in expressions for item in expression.imports):
        prefix, separator, names = statement.partition(" import ")
        if separator and prefix.startswith("from "):
            grouped.setdefault(prefix[5:], set()).update(name.strip() for name in names.split(","))
        else:
            other.add(statement)
    imports = [
        f"from {module} import {', '.join(sorted(names))}" for module, names in grouped.items()
    ]
    return tuple(sorted((*imports, *other)))


def run_list(registrations: LoadedRegistrations) -> int:
    """Print available Target values and the persistent entries behind them."""
    targets = available_targets()
    expressions = tuple(tgt.to_python() for tgt in targets)
    width = max((len(expression.text) for expression in expressions), default=0)
    lines = ["Available targets:", ""]
    lines.extend(
        f"  {expression.text:<{width}}  identity: {tgt.identity}"
        + ("   added" if tgt.identity in registrations.added_identities else "")
        for tgt, expression in zip(targets, expressions)
    )
    lines.extend(("", *merged_imports(expressions), ""))
    if registrations.state.documents or registrations.state.modules:
        lines.extend(("Added to this environment:",))
        lines.extend(
            f"  document  {entry.id:<28} {entry.source}"
            for entry in registrations.state.documents
        )
        lines.extend(
            f"  module    {entry.key:<28} {entry.name or entry.source}"
            for entry in registrations.state.modules
        )
        removable = (
            registrations.state.documents[0].id
            if registrations.state.documents
            else registrations.state.modules[0].key
        )
        lines.extend(("", f"  tilefoundry target remove {removable}", ""))
    lines.extend(
        (
            "What one of them states, and where each number came from:",
            "  tilefoundry target show <device id>",
        )
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def target_by_identity(identity: str) -> target.Target:
    """Find one available Target value by its exact identity."""
    targets = available_targets()
    matches = tuple(tgt for tgt in targets if tgt.identity == identity)
    if len(matches) == 1:
        return matches[0]
    available = sorted(tgt.identity for tgt in targets)
    if not matches:
        raise ValueError(f"unknown target identity {identity!r}; available: {available}")
    raise ValueError(f"target identity {identity!r} is ambiguous")


def format_document_target(tgt: target.Target) -> str:
    """Format the attributed hardware documents retained by one Target."""
    return hardware.format_capabilities(hardware.documents(tgt))


def run_show(identity: str) -> int:
    """Show one available Target by exact identity."""
    tgt = target_by_identity(identity)
    try:
        output = format_document_target(tgt)
    except hardware.UnknownDocumentError:
        output = f"identity: {tgt.identity}\n{tgt.to_python().text}\nfacts: unavailable"
    sys.stdout.write(output + "\n")
    return 0


def parse_source_document(source: str) -> tuple[Path, hardware.Document, type[target.Target]]:
    path = Path(source).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    document = hardware.parse_document(text, origin_label=str(path))
    return path, document, hardware_owner(document.schema)


def run_add_document(source: str, registrations: LoadedRegistrations) -> int:
    """Validate and persist one hardware document."""
    path, document, owner = parse_source_document(source)
    missing = missing_architecture(owner, document)
    if missing is not None:
        raise ValueError(
            f"document {document.id!r} needs architecture {missing!r}; add it first"
        )
    if document.kind == "device":
        occupied = identity_owners().get(document.id)
        if occupied is not None and document.id not in owner.hardware.documents():
            raise ValueError(
                f"target identity {document.id!r} is already occupied by "
                f"{occupied.to_python().text} ({type(occupied).__module__}.{type(occupied).__qualname__})"
            )
    owner.hardware.adopt(document)
    entry = DocumentEntry(document.id, path, document.digest)
    loaded_ids = {item.entry.id for item in registrations.documents}
    replacing = any(
        item.id == document.id and item.id not in loaded_ids
        for item in registrations.state.documents
    )
    if replacing:
        documents = tuple(
            entry if item.id == document.id else item
            for item in registrations.state.documents
        )
    else:
        documents = (*registrations.state.documents, entry)
    try:
        write_registry(
            registrations.path,
            RegistryState(documents, registrations.state.modules),
        )
    except Exception:
        owner.hardware.discard(document.id)
        raise
    if document.kind == "architecture":
        result = "architecture, no target of its own"
    else:
        result = owner(document.id).to_python().text
    sys.stdout.write(f"added  document  {document.id:<28} -> {result}\n")
    return 0


def module_entry(source: str) -> ModuleEntry:
    candidate = Path(source).expanduser()
    if source.endswith(".py") or candidate.is_file():
        path = candidate.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"target module file {path} does not exist")
        return ModuleEntry(source=path)
    return ModuleEntry(name=source)


def run_add_module(source: str, registrations: LoadedRegistrations) -> int:
    """Execute and persist one Target provider module."""
    entry = module_entry(source)
    if entry.source is not None:
        occupied = next(
            (
                item
                for item in registrations.state.modules
                if item.source is not None and item.key == entry.key
            ),
            None,
        )
        if occupied is not None:
            raise ValueError(
                f"module name {entry.key!r} is already added from {occupied.source}"
            )
    if entry in registrations.state.modules:
        raise ValueError(f"target module {entry.key!r} is already added")
    before = dict(target.registered())
    try:
        module = load_module(entry)
        target_types = module_targets(module, replay_cached=entry.name is not None)
        existing = identity_owners(frozenset(target_types))
        check_module_identities(target_types, existing)
        write_registry(
            registrations.path,
            RegistryState(registrations.state.documents, (*registrations.state.modules, entry)),
        )
        known_module_targets.remember(module.__name__, target_types)
    except Exception:
        rollback_targets(before)
        raise
    values = targets_of(target_types)
    rendered = ", ".join(tgt.to_python().text for tgt in values)
    sys.stdout.write(f"added  module    {entry.key:<28} -> {rendered}\n")
    if entry.source is not None:
        sys.stdout.write("note: this file will be executed on every tilefoundry run.\n")
    return 0


def run_remove(name: str, registrations: LoadedRegistrations) -> int:
    """Remove one document or provider module by any listed name."""
    documents = tuple(entry for entry in registrations.state.documents if entry.id == name)
    modules = tuple(
        entry
        for entry in registrations.state.modules
        if name == entry.key
        or any(
            loaded.entry == entry and name in loaded.identities
            for loaded in registrations.modules
        )
    )
    if len(documents) + len(modules) != 1:
        if not documents and not modules:
            available = [
                *(entry.id for entry in registrations.state.documents),
                *(entry.key for entry in registrations.state.modules),
                *(identity for loaded in registrations.modules for identity in loaded.identities),
            ]
            raise ValueError(f"no added target entry {name!r}; removable: {sorted(set(available))}")
        raise ValueError(f"added target entry {name!r} is ambiguous")

    if documents:
        entry = documents[0]
        loaded = next(
            (item for item in registrations.documents if item.entry == entry), None
        )
        if loaded is not None:
            loaded.owner.hardware.discard(entry.id)
        state = RegistryState(
            tuple(item for item in registrations.state.documents if item != entry),
            registrations.state.modules,
        )
        write_registry(registrations.path, state)
        sys.stdout.write(f"removed  document  {entry.id}\n")
        return 0

    entry = modules[0]
    loaded = next((item for item in registrations.modules if item.entry == entry), None)
    identities = () if loaded is None else loaded.identities
    if loaded is not None:
        for target_type in loaded.target_types:
            target.unregister(target_type)
    state = RegistryState(
        registrations.state.documents,
        tuple(item for item in registrations.state.modules if item != entry),
    )
    write_registry(registrations.path, state)
    sys.stdout.write(
        f"removed  module    {entry.key}; identities: {list(identities)}\n"
    )
    return 0


__all__ = [
    "LoadedRegistrations",
    "available_targets",
    "format_document_target",
    "load_registrations",
    "registry_path",
    "run_add_document",
    "run_add_module",
    "run_list",
    "run_remove",
    "run_show",
    "target_by_identity",
]
