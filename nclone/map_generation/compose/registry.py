"""Component / template registries + decorators + introspection.

This is the agentic surface: an agent calls :func:`list_components` /
:func:`describe` to enumerate building blocks and read each knob's meaning
without reading source, then assembles a template.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .component import Component, ParamRange
from . import labels as labels_mod

COMPONENT_REGISTRY: Dict[str, Component] = {}
TEMPLATE_REGISTRY: Dict[str, type] = {}


def register_component(name: str, tags, params: Optional[Dict[str, ParamRange]] = None,
                       ports: Optional[Dict[str, str]] = None,
                       difficulty: Optional[Dict[str, int]] = None,
                       min_w: int = 3, min_h: int = 2):
    """Class decorator: configure a :class:`Component` subclass and register an instance."""
    tags = set(tags)
    for t in tags:
        labels_mod.validate_tag(t)

    def deco(cls: Type[Component]) -> Type[Component]:
        cls.name = name
        cls.tags = tags
        cls.param_spec = params or {}
        if ports is not None:
            cls.ports = ports
        cls.difficulty = difficulty or {}
        cls.min_w = min_w
        cls.min_h = min_h
        COMPONENT_REGISTRY[name] = cls()
        return cls

    return deco


def register_template(name: str, **attrs):
    """Class decorator: register a :class:`Template` subclass under ``name``."""

    def deco(cls):
        cls.name = name
        for k, v in attrs.items():
            setattr(cls, k, v)
        TEMPLATE_REGISTRY[name] = cls
        return cls

    return deco


# --- introspection ------------------------------------------------------------
def list_components(tag: Optional[str] = None, domain: Optional[str] = None) -> List[str]:
    out = []
    for name, comp in COMPONENT_REGISTRY.items():
        if tag is not None and tag not in comp.tags:
            continue
        if domain is not None and not any(t.split(":", 1)[0] == domain for t in comp.tags):
            continue
        out.append(name)
    return sorted(out)


def list_templates() -> List[str]:
    return sorted(TEMPLATE_REGISTRY.keys())


def describe(name: str) -> Dict:
    comp = COMPONENT_REGISTRY[name]
    return {
        "name": comp.name,
        "tags": sorted(comp.tags),
        "mechanics": labels_mod.mechanics_of(comp.tags),
        "rollup_bucket": labels_mod.rollup_bucket(comp.tags),
        "params": {k: spec.describe() for k, spec in comp.param_spec.items()},
        "ports": dict(comp.ports),
        "difficulty": dict(comp.difficulty),
        "min_size": [comp.min_w, comp.min_h],
    }


def template_labels(name: str, seed: int = 0) -> Dict:
    """Build a template once and report its realized labels (a representative sample)."""
    cls = TEMPLATE_REGISTRY[name]
    _map, labels = cls().build(seed)
    return {
        "mechanics": list(labels.mechanics),
        "primary_bucket": labels.primary_bucket,
        "declared_difficulty": dict(labels.declared_difficulty),
    }


def template_possible_tags(name: str) -> set:
    """Union of every fine tag any component admitted by this template's cell
    filters can contribute -- the SEED-INDEPENDENT upper bound used for curriculum
    gating (so no random seed can sneak in a tag/difficulty above the ceiling).
    """
    cls = TEMPLATE_REGISTRY[name]
    tmpl = cls()
    tags: set = {"puzzle:exit_switch"}
    filters = list(getattr(tmpl, "cells", {}).values())
    explicit = getattr(tmpl, "explicit_components", None)
    if explicit:
        for cname in explicit.values():
            comp = COMPONENT_REGISTRY.get(cname)
            if comp:
                tags |= set(comp.tags)
    for filt in filters:
        for comp in COMPONENT_REGISTRY.values():
            try:
                if filt(comp):
                    tags |= set(comp.tags)
            except Exception:
                continue
    return tags


def list_templates_matching(allowed_tags, ceilings: Optional[Dict[str, int]] = None,
                            required_tag: Optional[str] = None) -> List[str]:
    """Label-filter selector for curriculum: templates whose POSSIBLE ``mechanics``
    (over every component their cell filters admit) are a subset of ``allowed_tags``
    (ignoring the always-present ``puzzle:exit_switch``), under per-domain
    ``ceilings``, optionally requiring a specific tag. This is the mechanism a
    curriculum stage uses to pull skill-targeted levels -- impossible with coarse
    buckets alone.
    """
    allowed = set(allowed_tags) | {"puzzle:exit_switch"}
    out = []
    for name in TEMPLATE_REGISTRY:
        possible = template_possible_tags(name)
        mech = set(labels_mod.mechanics_of(possible))
        if not mech <= allowed:
            continue
        if required_tag is not None and required_tag not in mech:
            continue
        if ceilings:
            diff = labels_mod.difficulty_vector(possible)
            if any(diff.get(dom, 0) > cap for dom, cap in ceilings.items()):
                continue
        out.append(name)
    return sorted(out)


def describe_template(name: str) -> Dict:
    cls = TEMPLATE_REGISTRY[name]
    return {
        "name": getattr(cls, "name", name),
        "grid": list(getattr(cls, "grid", (2, 3))),
        "grid_jitter": getattr(cls, "grid_jitter", False),
        "path": getattr(cls, "path", "monotone_walk"),
        "cells": {k: getattr(v, "__doc__", "filter") for k, v in
                  getattr(cls, "cells", {}).items()},
    }
