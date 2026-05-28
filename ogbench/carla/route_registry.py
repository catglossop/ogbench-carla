"""Map CARLA benchmark route XMLs to interpretable, command-line-friendly names.

Two sources are supported:

* **Bench2Drive** — 220 single-route XMLs under ``data/bench2drive_split/``
  (vendored into this repo). Names are derived from the ``<scenario type>``
  tag plus a per-type counter, e.g. ``parking-cut-in-001``.
* **Fail2Drive** — 200 single-route XMLs shipped by the ``fail2drive`` Python
  package (``catglossop/fail2drive`` fork). Names are derived from the file
  basename, e.g. ``base-pedestrians-on-road-0085`` / ``generalization-animals-1075``.
  Both the original filename (``Base_PedestriansOnRoad_0085``) and the
  kebab-cased form work as lookup keys.

Each XML holds **one** ``<route id="N">`` (Fail2Drive routes may carry zero or
one ``<scenario>`` tags; some long-tail scenarios are baked into world setup
rather than declared in XML).

Looking up by ``scenario_name``, ``file_name``, ``route_id``, or — for
Fail2Drive — the prefixed key ``f2d:<id>`` returns the same :class:`RouteEntry`.
(The ``f2d:`` prefix exists because Fail2Drive's ``route_id`` values are
small integers like ``"0"``, ``"1"``, ``"75"`` that would otherwise collide
across files or with bench2drive ids.)

Usage::

    from ogbench.carla.route_registry import find_route, list_routes

    entry = find_route("parking-cut-in-001")            # bench2drive
    entry = find_route("base-pedestrians-on-road-0085") # fail2drive
    entry = find_route("Base_PedestriansOnRoad_0085")   # also works
    entry.xml_path     # absolute path on disk
    entry.route_id     # "1711" (bench2drive) or "85" (fail2drive)
    entry.town         # "Town12" / "Town13"
    entry.source       # "bench2drive" or "fail2drive"

The registry is parsed lazily and memoised; the first call to any function in
this module walks both source directories once.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_DEFAULT_BENCH2DRIVE_DIR = Path(__file__).resolve().parents[2] / "data" / "bench2drive_split"


def _fail2drive_routes_dir() -> Optional[Path]:
    """Return the Fail2Drive routes directory.

    Resolution order:
    1. ``FAIL2DRIVE_ROUTES_DIR`` env var  (e.g. ~/fail2drive/fail2drive_split)
    2. ``fail2drive`` Python package's ``ROUTES_DIR`` attribute (if the package
       is pip-installed, e.g. catglossop/fail2drive)
    """
    # 1. Explicit env var — works with a cloned repo, no pip install needed.
    env_dir = os.environ.get("FAIL2DRIVE_ROUTES_DIR", "")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.is_dir():
            return p

    # 2. Installed package fallback.
    try:
        import fail2drive  # type: ignore
    except ImportError:
        return None
    routes_dir = Path(getattr(fail2drive, "ROUTES_DIR", ""))
    return routes_dir if routes_dir.is_dir() else None


@dataclass(frozen=True)
class RouteEntry:
    """One CARLA benchmark route + scenario."""

    scenario_name: str       # e.g. "parking-cut-in-001" or "base-pedestrians-on-road-0085"
    file_name: str           # e.g. "bench2drive_007" or "Base_PedestriansOnRoad_0085"
    xml_path: Path           # absolute path to the XML
    route_id: str            # the <route id="..."> string (matches leaderboard's --routes-subset)
    scenario_type: str       # CamelCase type from <scenario type="..."> (empty string if none declared)
    scenario_subname: str    # the <scenario name="..."> attribute (empty string if none declared)
    town: str                # e.g. "Town12", "Town13"
    source: str              # "bench2drive" or "fail2drive"


def _kebab(camel: str) -> str:
    """Convert ``ParkingCutIn`` -> ``parking-cut-in`` (handles acronyms reasonably)."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", camel)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1)
    return s2.lower()


def _parse_one(xml_path: Path) -> Optional[Tuple[str, str, str, str]]:
    """Return ``(route_id, scenario_type, scenario_subname, town)`` for ``xml_path``.

    Returns ``None`` if the file lacks a ``<route>``. ``scenario_type`` /
    ``scenario_subname`` may be empty strings when the file has an empty
    ``<scenarios/>`` (Fail2Drive's world-setup-only routes).
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    routes = list(root.iter("route"))
    if not routes:
        return None
    route = routes[0]
    route_id = str(route.attrib.get("id", "")).strip()
    town = str(route.attrib.get("town", "")).strip()
    if not (route_id and town):
        return None
    scenarios = list(route.iter("scenario"))
    if scenarios:
        sc = scenarios[0]
        scenario_type = str(sc.attrib.get("type", "")).strip()
        scenario_subname = str(sc.attrib.get("name", "")).strip()
    else:
        scenario_type = ""
        scenario_subname = ""
    return route_id, scenario_type, scenario_subname, town


def _add_entry(by_key: Dict[str, RouteEntry], key: str, entry: RouteEntry) -> None:
    existing = by_key.get(key)
    if existing is not None and existing is not entry:
        raise ValueError(
            f"Duplicate route key {key!r}: {existing.xml_path} vs {entry.xml_path}"
        )
    by_key[key] = entry


def _scan_bench2drive(routes_dir: Path, by_key: Dict[str, RouteEntry]) -> List[RouteEntry]:
    """Add all bench2drive entries to ``by_key`` and return them in scan order."""
    files = sorted(routes_dir.glob("*.xml"))
    counters: Dict[str, int] = {}
    entries: List[RouteEntry] = []
    for xml_path in files:
        info = _parse_one(xml_path)
        if info is None:
            continue
        route_id, scenario_type, scenario_subname, town = info
        if not scenario_type:
            # bench2drive XMLs always declare a scenario; skip if not
            continue
        slug = _kebab(scenario_type)
        counters[slug] = counters.get(slug, 0) + 1
        scenario_name = f"{slug}-{counters[slug]:03d}"
        file_name = xml_path.stem
        entry = RouteEntry(
            scenario_name=scenario_name,
            file_name=file_name,
            xml_path=xml_path.resolve(),
            route_id=route_id,
            scenario_type=scenario_type,
            scenario_subname=scenario_subname,
            town=town,
            source="bench2drive",
        )
        entries.append(entry)
        for key in (scenario_name, file_name, route_id):
            _add_entry(by_key, key, entry)
    return entries


_FAIL2DRIVE_NAME_RE = re.compile(r"^(Base|Generalization)_([A-Za-z]+)_(\d+)$")


def _scan_fail2drive(routes_dir: Path, by_key: Dict[str, RouteEntry]) -> List[RouteEntry]:
    """Add all fail2drive entries to ``by_key`` and return them in scan order."""
    files = sorted(routes_dir.glob("*.xml"))
    entries: List[RouteEntry] = []
    for xml_path in files:
        info = _parse_one(xml_path)
        if info is None:
            continue
        route_id, scenario_type, scenario_subname, town = info
        file_name = xml_path.stem
        m = _FAIL2DRIVE_NAME_RE.match(file_name)
        if m:
            split, raw_kind, idx = m.group(1), m.group(2), m.group(3)
            scenario_name = f"{split.lower()}-{_kebab(raw_kind)}-{idx}"
        else:
            scenario_name = _kebab(file_name)
        entry = RouteEntry(
            scenario_name=scenario_name,
            file_name=file_name,
            xml_path=xml_path.resolve(),
            route_id=route_id,
            scenario_type=scenario_type,
            scenario_subname=scenario_subname,
            town=town,
            source="fail2drive",
        )
        entries.append(entry)
        # Fail2Drive route_ids are small integers (0-99, 1000-1099) that would
        # collide with each other across files and could collide with
        # bench2drive ids. We expose them only via a prefixed key.
        for key in (scenario_name, file_name, f"f2d:{route_id}"):
            _add_entry(by_key, key, entry)
    return entries


@lru_cache(maxsize=8)
def _build_registry(
    bench2drive_dir_str: str,
    fail2drive_dir_str: str,
) -> Dict[str, RouteEntry]:
    """Walk both source dirs once; build key -> entry dict."""
    by_key: Dict[str, RouteEntry] = {}

    b2d_dir = Path(bench2drive_dir_str)
    if b2d_dir.is_dir():
        _scan_bench2drive(b2d_dir, by_key)
    elif bench2drive_dir_str:
        # An explicit non-empty path that doesn't exist is a config error.
        raise FileNotFoundError(f"Bench2Drive routes dir not found: {b2d_dir}")

    f2d_dir = Path(fail2drive_dir_str) if fail2drive_dir_str else None
    if f2d_dir and f2d_dir.is_dir():
        _scan_fail2drive(f2d_dir, by_key)

    if not by_key:
        raise FileNotFoundError(
            "No route sources available: pass routes_dir= or install fail2drive."
        )
    return by_key


def get_routes_dir(override: Optional[Path] = None) -> Path:
    """Return the bench2drive routes dir (kept for backwards compatibility)."""
    return Path(override) if override else _DEFAULT_BENCH2DRIVE_DIR


def _resolve_source_dirs(
    routes_dir: Optional[Path],
    fail2drive_dir: Optional[Path],
) -> Tuple[str, str]:
    b2d = str(get_routes_dir(routes_dir))
    f2d = Path(fail2drive_dir) if fail2drive_dir else _fail2drive_routes_dir()
    return b2d, str(f2d) if f2d else ""


def find_route(
    name: str,
    routes_dir: Optional[Path] = None,
    fail2drive_dir: Optional[Path] = None,
) -> RouteEntry:
    """Resolve a route by scenario name, file basename, or numeric route id.

    For Fail2Drive routes whose numeric ``route_id`` would otherwise collide
    (small integers reused across files), pass ``f2d:<id>`` instead.

    Raises ``KeyError`` with a hint listing close matches when ``name`` is unknown.
    """
    registry = _build_registry(*_resolve_source_dirs(routes_dir, fail2drive_dir))
    key = name.strip()
    if key in registry:
        return registry[key]
    lk = key.lower()
    if lk in registry:
        return registry[lk]
    near = [k for k in registry if isinstance(k, str) and lk in k.lower()][:5]
    hint = f" Did you mean one of: {near}?" if near else ""
    raise KeyError(f"Route {name!r} not in registry.{hint}")


def list_routes(
    routes_dir: Optional[Path] = None,
    fail2drive_dir: Optional[Path] = None,
    source: Optional[str] = None,
) -> List[RouteEntry]:
    """Return all registered route entries (each route appears once).

    ``source`` filters to ``"bench2drive"`` or ``"fail2drive"`` if provided.
    """
    registry = _build_registry(*_resolve_source_dirs(routes_dir, fail2drive_dir))
    seen: List[RouteEntry] = []
    seen_paths = set()
    for entry in registry.values():
        if entry.xml_path in seen_paths:
            continue
        if source is not None and entry.source != source:
            continue
        seen.append(entry)
        seen_paths.add(entry.xml_path)
    return sorted(seen, key=lambda e: (e.source, e.scenario_name))


def list_scenario_types(
    routes_dir: Optional[Path] = None,
    fail2drive_dir: Optional[Path] = None,
) -> List[str]:
    return sorted(
        {e.scenario_type for e in list_routes(routes_dir, fail2drive_dir) if e.scenario_type}
    )
