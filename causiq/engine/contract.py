"""
Contract loader — the single governed entry point to KPI semantics.

Nothing else in the engine may hard-code a formula, a threshold, an adjustment
set or an access rule. If it is a business decision, it lives in YAML and is
read through here. This is what makes the prototype auditable: a governance
reviewer can diff the contract without reading Python.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONTRACT_DIR = Path(__file__).resolve().parent.parent / "contracts"


@functools.lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    return yaml.safe_load((CONTRACT_DIR / "kpi_contract.yaml").read_text())


@functools.lru_cache(maxsize=1)
def load_levers() -> dict[str, Any]:
    return yaml.safe_load((CONTRACT_DIR / "lever_library.yaml").read_text())


# -----------------------------------------------------------------------------
# Typed accessors
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class KpiSpec:
    name: str
    label: str
    definition: str
    unit: str
    formula: str | None
    children: list[str]
    sources: list[str]
    baseline_method: str
    materiality: dict[str, Any]
    lineage: list[str]
    sensitive: bool
    mixed_source_ratio: bool


def kpi(name: str) -> KpiSpec:
    raw = load_contract()["kpis"][name]
    return KpiSpec(
        name=name,
        label=raw["label"],
        definition=raw["definition"].strip(),
        unit=raw["unit"],
        formula=raw.get("formula"),
        children=raw.get("children") or [],
        sources=raw["sources"],
        baseline_method=raw["baseline_method"],
        materiality=raw["materiality"],
        lineage=raw.get("lineage", []),
        sensitive=bool(raw.get("sensitive", False)),
        mixed_source_ratio=bool(raw.get("mixed_source_ratio", False)),
    )


def all_kpis() -> list[str]:
    return list(load_contract()["kpis"].keys())


def source(name: str) -> dict[str, Any]:
    return load_contract()["sources"][name]


def driver(name: str) -> dict[str, Any]:
    return load_contract()["drivers"][name]


def all_drivers() -> list[str]:
    return list(load_contract()["drivers"].keys())


def identification(driver_name: str) -> dict[str, Any] | None:
    """How this driver's causal effect may be identified.

    Returning None means the contract does not sanction a causal claim for this
    driver. The engine must then report it as an association, never a cause.
    """
    return load_contract()["causal_dag"]["identification"].get(driver_name)


def dag_edges() -> list[tuple[str, str]]:
    return [tuple(e) for e in load_contract()["causal_dag"]["edges"]]


def mediators_of(target: str) -> set[str]:
    """Nodes that sit on a path INTO `target` and are themselves caused by a driver.

    Used to prevent double counting: a variable that is both a parent of the KPI
    and a child of another driver is a mediator, not an independent cause.
    """
    edges = dag_edges()
    parents_of_target = {p for p, c in edges if c == target}
    caused = {c for _, c in edges}
    return {n for n in parents_of_target if n in caused}


def confidence_config() -> dict[str, Any]:
    return load_contract()["confidence"]


def sparse_config() -> dict[str, Any]:
    return load_contract()["sparse_history"]


def entitlement(role: str) -> dict[str, Any]:
    ents = load_contract()["entitlements"]["roles"]
    if role not in ents:
        raise PermissionError(f"Unknown role '{role}' — access denied by default.")
    return ents[role]


def persona(key: str) -> dict[str, Any]:
    return load_contract()["personas"][key]


def all_personas() -> list[str]:
    return list(load_contract()["personas"].keys())


def levers_for(driver_name: str) -> list[dict[str, Any]]:
    return [lv for lv in load_levers()["levers"] if lv["driver"] == driver_name]


def routing_for(driver_name: str) -> dict[str, Any] | None:
    return load_levers().get("routing", {}).get(driver_name)
