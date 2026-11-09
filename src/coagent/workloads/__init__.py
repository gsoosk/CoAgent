"""Contended-cell registry."""

from .base import CELLS, Cell, invariant_ok, normalize, register
from . import k8s          # noqa: F401  (registers the aiopslab cells)
from . import workbench   # noqa: F401  (registers the workbench cells)
from . import chopping_cells  # noqa: F401  (idea.pdf workloads)

# The ten cells built to race. The chopping-paper workloads (suite
# "chopping") are throughput and admission structures, not contention
# cells -- cc-claim is deliberately conflict-light, which is the point.
CONTENDED = {n: c for n, c in CELLS.items()
             if c.suite in ("aiopslab", "workbench")}

__all__ = ["CELLS", "CONTENDED", "Cell", "invariant_ok", "normalize",
           "register"]
