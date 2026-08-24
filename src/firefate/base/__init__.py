"""Contracts and primitives shared by all three FIREFate modules.

Nothing here is domain-specific. :func:`calculate_tf_episodic_enrichment` is the
single over-representation primitive both the Temporal and StateSpecific modules
run; :class:`BaseManager` is the bookkeeping every module manager inherits.
"""

from firefate.base.enrichment import calculate_tf_episodic_enrichment
from firefate.base.manager import BaseManager

#: Readable alias for the over-representation primitive.
ora = calculate_tf_episodic_enrichment

__all__ = ["BaseManager", "calculate_tf_episodic_enrichment", "ora"]
