"""Tests for partition module.

Consolidated from:
- inspect_kahip.py (inspect kahip API)
- inspect_pymetis.py (inspect pymetis API)
- test_cyclic_expansion.py (cyclic expansion refinement test)
"""

import inspect
import importlib
import json
import numpy as np
import pytest


# ── kahip inspection ────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not importlib.util.find_spec("kahip"),
    reason="kahip is not installed",
)
class TestKahipInspection:
    """Inspect the kahip.kaffpa API signature."""

    def test_kaffpa_signature(self):
        import kahip
        sig = inspect.signature(kahip.kaffpa)
        assert sig is not None
        params = list(sig.parameters.keys())
        assert len(params) > 0, "kaffpa should accept parameters"

    def test_kaffpa_attrs(self):
        import kahip
        attrs = [a for a in dir(kahip) if 'kaffpa' in a.lower() or 'part' in a.lower()]
        assert len(attrs) > 0, "kahip should have kaffpa-related attributes"


# ── pymetis inspection ─────────────────────────────────────────────────────

@pytest.mark.skipif(
    not importlib.util.find_spec("pymetis"),
    reason="pymetis is not installed",
)
class TestPymetisInspection:
    """Inspect the pymetis.part_graph API signature."""

    def test_part_graph_signature(self):
        pymetis = importlib.import_module('pymetis')
        sig = inspect.signature(pymetis.part_graph)
        assert sig is not None
        params = list(sig.parameters.keys())
        assert len(params) > 0, "part_graph should accept parameters"


# ── Cyclic expansion refinement test ───────────────────────────────────────

def _compute_cut(adj, parts):
    n = len(parts)
    cut = 0.0
    for i in range(n):
        for j, w in adj[i]:
            if i < j and parts[i] != parts[j]:
                cut += w
    return cut


class TestCyclicExpansion:
    """Test cyclic expansion refinement on a small graph."""

    def test_cyclic_expansion_on_toy_graph(self):
        from src.fem.cyclic_expansion import cyclic_expansion_refine

        adj = [[] for _ in range(6)]
        edges = [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
            (2, 3, 0.5),
        ]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))

        partition = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        q = 2

        new_part = cyclic_expansion_refine(
            adjacency=adj,
            partition=partition,
            q=q,
            max_iterations=10,
            max_candidates=10,
            num_trials=4,
            num_steps=100,
            dev='cpu',
            patience=3,
            verbose=False,
        )

        initial_cut = _compute_cut(adj, partition)
        refined_cut = _compute_cut(adj, new_part)

        # Refinement should not increase the cut
        assert refined_cut <= initial_cut + 1e-9, (
            f"Refinement increased cut from {initial_cut} to {refined_cut}"
        )
        # Both parts should still be valid labels
        assert set(new_part) == {0, 1}, "Labels should be 0 and 1"
