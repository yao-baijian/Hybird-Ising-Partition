"""Hybrid Ising Partition — src package root.

Solvers (FEM, SBM, QIS3) are provided by the ``qubo-solver`` submodule
at ``lib/qubo-solver/``.  This ``__init__`` adds it to the module search
path so shim imports in ``src/fem/``, ``src/sbm/`` etc. work transparently.
"""

import sys
from pathlib import Path

_QUBO_SOLVER_PATH = Path(__file__).resolve().parents[1] / "lib" / "qubo-solver"
if _QUBO_SOLVER_PATH.is_dir() and str(_QUBO_SOLVER_PATH) not in sys.path:
    sys.path.insert(0, str(_QUBO_SOLVER_PATH))
