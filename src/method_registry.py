"""
Partition method registry — central registry for all partition pipeline methods.

Each method is a combination of an initial-partition solver and a refinement
solver.  Methods are registered with a unique name, a MethodName (family +
algorithm), a human-readable description, and default parameter values.

Usage:
    from src.partition.method_registry import registry, MethodName, PartitionMethod

    # Look up a method
    method = registry['init_fem_refine_metis']
    result = method.run(J, q, **overrides)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class MethodName:
    """Two-level method name: family (pipeline) + algorithm (solver)."""
    def __init__(self, family: str, algorithm: str):
        self.family = family
        self.algorithm = algorithm

    def __str__(self):
        return f'{self.family}: {self.algorithm}'

    def __repr__(self):
        return f'{self.family}: {self.algorithm}'

    def __eq__(self, other):
        if isinstance(other, MethodName):
            return self.family == other.family and self.algorithm == other.algorithm
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


class PartitionMethod:
    """Descriptor for a single partition pipeline method."""

    def __init__(
        self,
        name: str,
        method_name: MethodName,
        description: str,
        run_func: Callable,
        solver_names: Optional[list[str]] = None,
    ):
        self.name = name
        self.method_name = method_name
        self.description = description
        self.run_func = run_func
        self.solver_names = solver_names or []

    def run(self, J, q, config_dir=None, **kwargs):
        """Execute this method by merging solver configs (config_dir) with kwargs."""
        params = {}
        for sn in self.solver_names:
            solver_cfg = load_config(sn, config_dir) if config_dir else {}
            params.update(solver_cfg)
        params.update(kwargs)
        return self.run_func(J, q, **params)

    def __repr__(self):
        return f"<PartitionMethod {self.name}: {self.method_name}>"


class _Registry:
    """Global method registry (singleton-like)."""

    def __init__(self):
        self._methods: Dict[str, PartitionMethod] = {}

    def register(
        self,
        name: str,
        family: str,
        algorithm: str,
        description: str = "",
        run_func: Optional[Callable] = None,
        solver_names: Optional[list[str]] = None,
    ):
        """Register a method.  run_func can be set later via .bind()."""
        if name in self._methods:
            raise KeyError(f"Method '{name}' already registered")
        mn = MethodName(family, algorithm)
        self._methods[name] = PartitionMethod(
            name=name,
            method_name=mn,
            description=description,
            run_func=run_func,
            solver_names=solver_names,
        )
        return self._methods[name]

    def bind(self, name: str, run_func: Callable):
        """Attach (or replace) the run function for an already-registered method."""
        if name not in self._methods:
            raise KeyError(f"Method '{name}' not yet registered — call register() first")
        self._methods[name].run_func = run_func

    def lookup(self, name: str) -> PartitionMethod:
        if name not in self._methods:
            available = ", ".join(sorted(self._methods))
            raise KeyError(f"Unknown method '{name}'. Available: {available}")
        return self._methods[name]

    def __getitem__(self, name: str) -> PartitionMethod:
        return self.lookup(name)

    def __contains__(self, name: str) -> bool:
        return name in self._methods

    def keys(self):
        return self._methods.keys()

    def items(self):
        return self._methods.items()

    def values(self):
        return self._methods.values()

    def __iter__(self):
        return iter(self._methods.values())

    def __len__(self):
        return len(self._methods)


# Global singleton
registry = _Registry()


# ── Per-solver JSON config helpers ────────────────────────────────────────
#   Each solver (fem, sbm, kaffpa, metis, cyclic) has a default JSON under
#   src/partition/configs/{solver}.json.  At test time these are copied to
#   the working config/ directory (gitignored) where users can override them.

CONFIG_SRC = Path(__file__).resolve().parent / "configs"
CONFIG_DST = Path.cwd() / "config"  # gitignored working copy


def ensure_configs(copy_to: Optional[Path] = None):
    """Copy default solver JSONs from src/ to working config/ if absent."""
    dst = copy_to or CONFIG_DST
    dst.mkdir(parents=True, exist_ok=True)
    if CONFIG_SRC.is_dir():
        for f in sorted(CONFIG_SRC.glob("*.json")):
            target = dst / f.name
            if not target.exists():
                import shutil
                shutil.copy2(str(f), str(target))
                print(f"[config] Created default: {target}")


def load_config(solver_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load a single solver's config from the working config directory."""
    cfg_dir = config_dir or CONFIG_DST
    cfg_file = cfg_dir / f"{solver_name}.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            return json.load(f)
    return {}


def merge_config(method_name: str, overrides: Dict[str, Any],
                 config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load all solver configs for a registered method and overlay overrides."""
    if method_name not in registry:
        return overrides
    method = registry[method_name]
    params = {}
    for sn in method.solver_names:
        solver_cfg = load_config(sn, config_dir)
        params.update(solver_cfg)
    params.update(overrides)
    return params
