"""MindHier scale-aware hierarchy-to-hierarchy autoregressive transformer.

The original research code used the name ``Switti`` in several internals.  The
public Stage 2 model name is ``HierarchyAR``.  Backward-compatible aliases are
kept in ``models.switti`` so older checkpoints and notebooks can still load.
"""

from models.switti import HierarchyAR, HierarchyARHF, get_crop_condition

__all__ = ["HierarchyAR", "HierarchyARHF", "get_crop_condition"]
