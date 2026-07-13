"""
Shared ontology loader for the Nexus pipeline (Steps 3, 3.5, 4).

Generalization goal: one codebase, many regulators. The domain taxonomy and the
canonical concept catalog are LAYERED — a jurisdiction-neutral GLOBAL core plus
a per-regulator extension, merged at runtime by jurisdiction. Universal
regulatory primitives (obligation types, actions, modality) live only in the
global layer, since "Reporting" or "RETAIN" mean the same thing everywhere.

Config layout (under <config-root>/ontology/):
    global/taxonomies.yaml   obligation_type, action, mandatory_flag, escape_hatch
    global/domains.yaml      jurisdiction-neutral domains (+ compact tokens)
    global/concepts.yaml     jurisdiction-neutral canonical concepts
    jurisdictions/registry.yaml         regulator string -> pack code, + default
    jurisdictions/<CODE>/domains.yaml   extra/override domains
    jurisdictions/<CODE>/concepts.yaml  extra/override concepts

Nothing here is hardcoded to one deployment: the config root is resolved by
precedence (explicit arg > NEXUS_CONFIG_DIR env > <nexus>/config > packaged
default), and the jurisdiction is derived from the document's own Step-1
metadata (or forced with an explicit code).
"""

import json
import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parent / "Nexus" / "config"


def resolve_config_root(explicit: str | None, nexus_dir: Path | str) -> Path:
    """Locate the config root without hardcoding it. Precedence:
    --config value > NEXUS_CONFIG_DIR env > <nexus_dir>/config > packaged default."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("NEXUS_CONFIG_DIR")
    if env:
        return Path(env)
    candidate = Path(nexus_dir) / "config"
    if candidate.is_dir():
        return candidate
    return DEFAULT_CONFIG_ROOT


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_jurisdiction(config_root: Path, nexus_dir: Path | str, stem: str,
                         explicit: str | None = None) -> str | None:
    """Pick the jurisdiction pack. An explicit code wins; otherwise derive it
    from the Step-1 metadata `regulator` via jurisdictions/registry.yaml; else
    fall back to the registry `default` (may be None -> global-only)."""
    registry = {}
    reg_path = config_root / "ontology" / "jurisdictions" / "registry.yaml"
    if reg_path.exists():
        registry = _read_yaml(reg_path) or {}
    packs = registry.get("regulators", {}) or {}

    if explicit:
        return explicit

    meta_path = Path(nexus_dir) / "metadata" / f"{stem}_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                regulator = (json.load(f).get("regulator") or "").strip()
            if regulator in packs:
                return packs[regulator]
        except (json.JSONDecodeError, OSError):
            pass
    return registry.get("default")


def _labels(items: list) -> tuple:
    return tuple(i["label"] if isinstance(i, dict) else i for i in items)


def _merge_by_key(base: list[dict], extra: list[dict], key: str) -> list[dict]:
    """Union base + extra; an extra entry with the same key overrides the base
    one (so a jurisdiction can both add to and refine the global layer)."""
    merged = {item[key]: item for item in base}
    for item in extra:
        merged[item[key]] = item
    return list(merged.values())


def load_ontology(config_root: Path, jurisdiction: str | None = None) -> dict:
    """Load and merge the global ontology with one jurisdiction extension.

    Returns a dict with the merged domain taxonomy (+ token maps), the global
    obligation/action/modality taxonomies, the escape hatch, and the merged
    concept catalog. Raises SystemExit with an actionable message on any missing
    file or unknown jurisdiction.
    """
    onto_dir = Path(config_root) / "ontology"
    g = onto_dir / "global"
    try:
        tax = _read_yaml(g / "taxonomies.yaml")
        gdom = _read_yaml(g / "domains.yaml")
        gcon = _read_yaml(g / "concepts.yaml")
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Global ontology missing: {exc.filename}\n"
            f"Expected taxonomies.yaml, domains.yaml and concepts.yaml under {g}."
        )

    domains = list(gdom["domains"])
    concepts = list(gcon["concepts"])
    versions = {"global": gdom.get("version")}

    if jurisdiction:
        j = onto_dir / "jurisdictions" / jurisdiction
        if not j.is_dir():
            raise SystemExit(
                f"Unknown jurisdiction pack '{jurisdiction}' (expected {j}). "
                f"Add it under {onto_dir / 'jurisdictions'} or pass a known code."
            )
        jdom_path, jcon_path = j / "domains.yaml", j / "concepts.yaml"
        if jdom_path.exists():
            jdom = _read_yaml(jdom_path)
            domains = _merge_by_key(domains, jdom.get("domains", []) or [], key="token")
            versions[jurisdiction] = jdom.get("version")
        if jcon_path.exists():
            jcon = _read_yaml(jcon_path)
            concepts = _merge_by_key(concepts, jcon.get("concepts", []) or [], key="canonical_id")

    return {
        "jurisdiction": jurisdiction,
        "domains": domains,
        "domain_labels": tuple(d["label"] for d in domains),
        "token_to_label": {d["token"]: d["label"] for d in domains},
        "label_to_token": {d["label"]: d["token"] for d in domains},
        "obligation_type": _labels(tax["obligation_type"]),
        "action": _labels(tax["action"]),
        "mandatory_flag": _labels(tax["mandatory_flag"]),
        "escape_hatch": tax["escape_hatch"],
        "concepts": concepts,
        "versions": versions,
    }
