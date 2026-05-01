Seven capabilities
==================

FIREFate is organized around seven capabilities. They are the scientific contract for the
framework: the README and API are aligned with this list.

1. **Discover cellular programs (CPs)** underlying contrasting cell states using interpretable
   ML, and characterize their phenotypic roles (for example via gene-set enrichment).

2. **Construct state-specific GRNs** from single-cell multi-omics or scRNA-seq, plus a combined
   cross-state GRN whose connectivity spans both states (matching the scope of the programs from
   capability 1).

3. **Construct transition-window-specific GRNs** from scRNA and scATAC (matched or unmatched),
   and cluster regulatory edges to temporally order waves of TF regulation.

4. **Construct episodic GRNs** by retaining TF–target forces that stay temporally invariant across
   an episode **and** fall in the top-percentile tails of activating or repressive edges.

5. **Quantify enrichment of dynamic TF activity** within each CP using state-specific or episodic
   GRN connectivity (overrepresentation / hypergeometric enrichment of program genes among
   downstream targets).

6. **Quantify in-silico perturbation effects** from the phenotypic shift induced by perturbing
   enriched TFs within each CP.

7. **Stratify uncommitted populations for fate-bias** using CPs inferred from intervention (for
   example gene knockout) versus unperturbed control populations.

**FIREFate-NP (Network Prioritization)** refers to the downstream workflow that combines static and
dynamic GRNs with interpretable ML to obtain sparse gene sets (CPs) that distinguish cell fates,
then embeds them in high-resolution GRNs to surface TF-centric regulons.

Implementation note
-------------------

The pip-installable package under ``src/firefate/`` is growing toward full coverage of all seven
capabilities. Episodic dynamics, smoothed GRN curves, enrichment orchestration, and state-specific
GRN helpers are represented in the current API; see :doc:`reference` for module-level detail.
