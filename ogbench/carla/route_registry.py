"""Map Bench2Drive route XMLs to interpretable, command-line-friendly names.

Each file under ``data/bench2drive_split/`` (220 of them) holds **one** ``<route id="N">``
with **one** ``<scenario type="...">``. We expose three name flavors per route:

* ``scenario_name``   - kebab-cased ``type`` plus a per-type counter, e.g.
                        ``parking-cut-in-001``, ``opposite-vehicle-running-red-light-003``.
* ``file_name``       - the basename without ``.xml``, e.g. ``bench2drive_007``.
* ``route_id``        - the integer ``id`` attribute on ``<route>`` (also accepted as a string).

Looking up by any of those three returns the same :class:`RouteEntry`.

Usage::

    from ogbench.carla.route_registry import find_route, list_routes

    entry = find_route("parking-cut-in-001")
    entry.xml_path     # /home/carla/ogbench-carla/data/bench2drive_split/bench2drive_000.xml
    entry.route_id     # "1711"
    entry.town         # "Town12"
    entry.scenario_type  # "ParkingCutIn"

The registry is parsed lazily and memoised; the first call to any function in this
module walks the XML directory once.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_DEFAULT_ROUTES_DIR = Path(__file__).resolve().parents[2] / "data" / "bench2drive_split"


@dataclass(frozen=True)
class RouteEntry:
    """One Bench2Drive route + scenario."""

    scenario_name: str       # e.g. "parking-cut-in-001"
    file_name: str           # e.g. "bench2drive_007"
    xml_path: Path           # absolute path to the XML
    route_id: str            # the <route id="..."> string (matches leaderboard's --routes-subset)
    scenario_type: str       # CamelCase type from <scenario type="...">
    scenario_subname: str    # the <scenario name="..."> attribute (e.g. "ParkingCutIn_1")
    town: str                # e.g. "Town12"


def _kebab(camel: str) -> str:
    """Convert ``ParkingCutIn`` -> ``parking-cut-in`` (handles acronyms reasonably)."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", camel)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1)
    return s2.lower()


def _parse_one(xml_path: Path) -> Optional[Tuple[str, str, str, str]]:
    """Return ``(route_id, scenario_type, scenario_subname, town)`` for ``xml_path``.

    Returns ``None`` if the file lacks a ``<route>`` or its single ``<scenario>``.
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
    scenarios = list(route.iter("scenario"))
    if not scenarios:
        return None
    sc = scenarios[0]
    scenario_type = str(sc.attrib.get("type", "")).strip()
    scenario_subname = str(sc.attrib.get("name", "")).strip()
    if not (route_id and town and scenario_type):
        return None
    return route_id, scenario_type, scenario_subname, town


@lru_cache(maxsize=8)
def _build_registry(routes_dir_str: str) -> Dict[str, RouteEntry]:
    """Walk ``routes_dir`` once; build name -> entry dict with all three lookup keys."""
    routes_dir = Path(routes_dir_str)
    if not routes_dir.is_dir():
        raise FileNotFoundError(f"Bench2Drive routes dir not found: {routes_dir}")

    files = sorted(routes_dir.glob("*.xml"))
    parsed: List[Tuple[Path, Tuple[str, str, str, str]]] = []
    for xml_path in files:
        info = _parse_one(xml_path)
        if info is not None:
            parsed.append((xml_path, info))

    counters: Dict[str, int] = {}
    by_key: Dict[str, RouteEntry] = {}
    for xml_path, (route_id, scenario_type, scenario_subname, town) in parsed:
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
        )
        for key in (scenario_name, file_name, route_id):
            if key in by_key and by_key[key] is not entry:
                raise ValueError(
                    f"Duplicate route key {key!r}: {by_key[key].xml_path} vs {xml_path}"
                )
            by_key[key] = entry
    return by_key


def get_routes_dir(override: Optional[Path] = None) -> Path:
    return Path(override) if override else _DEFAULT_ROUTES_DIR


def find_route(name: str, routes_dir: Optional[Path] = None) -> RouteEntry:
    """Resolve a route by scenario name, file basename, or numeric route id.

    Raises ``KeyError`` with a hint listing close matches when ``name`` is unknown.
    """
    registry = _build_registry(str(get_routes_dir(routes_dir)))
    key = name.strip()
    if key in registry:
        return registry[key]
    lk = key.lower()
    if lk in registry:
        return registry[lk]
    near = [k for k in registry if isinstance(k, str) and lk in k.lower()][:5]
    hint = f" Did you mean one of: {near}?" if near else ""
    raise KeyError(f"Route {name!r} not in registry.{hint}")


def list_routes(routes_dir: Optional[Path] = None) -> List[RouteEntry]:
    """Return all registered route entries (each route appears once)."""
    registry = _build_registry(str(get_routes_dir(routes_dir)))
    seen: List[RouteEntry] = []
    seen_paths = set()
    for entry in registry.values():
        if entry.xml_path in seen_paths:
            continue
        seen.append(entry)
        seen_paths.add(entry.xml_path)
    return sorted(seen, key=lambda e: e.scenario_name)


def list_scenario_types(routes_dir: Optional[Path] = None) -> List[str]:
    return sorted({e.scenario_type for e in list_routes(routes_dir)})
