"""Temporal clustering of regulatory links into phases of regulation."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from firefate.temporal._align import AlignTimeScales
from scipy.stats import mannwhitneyu


def get_max_points(force_curves, dtime, top_k=5, temperature=1.0):
    """
    Find pseudotime points with the highest absolute force values using softmax weighting.

    Parameters:
    -----------
    force_curves : DataFrame
        Multi-indexed DataFrame with (TF, Target) as rows and time points as columns
    dtime : array-like
        Actual pseudotime values corresponding to each column/window
    top_k : int
        Number of top points to return per TF-Target pair
    temperature : float
        Temperature parameter for softmax (lower = more peaked distribution)

    Returns:
    --------
    list of dicts: Each containing TF, Target, pseudotime, window_idx, force value, and softmax_weight
    """
    max_points = []

    # Convert dtime to array if needed
    dtime = np.array(dtime)

    # Get unique TF-Target pairs
    for tf_target in force_curves.index:
        # Get the time series for this TF-Target pair
        time_series = force_curves.loc[tf_target]

        # Get absolute force values
        abs_forces = time_series.abs()

        # Apply softmax to absolute forces
        softmax_weights = np.exp(abs_forces / temperature) / np.sum(np.exp(abs_forces / temperature))

        # Get top_k indices with highest softmax weights
        top_indices = softmax_weights.nlargest(top_k).index

        for time_col in top_indices:
            # Get the window index (column position)
            window_idx = force_curves.columns.get_loc(time_col) if time_col in force_curves.columns else time_col

            max_points.append({
                'TF': tf_target[0],
                'Target': tf_target[1],
                'window_idx': window_idx,
                'pseudotime': dtime[window_idx],
                'force': time_series[time_col],
                'abs_force': abs_forces[time_col],
                'softmax_weight': softmax_weights[time_col]
            })

    return max_points


def aggregate_max_points(max_points, method='weighted_mean'):
    """
    Aggregate max points to get a single pseudotime per TF-Target regulation.

    Parameters:
    -----------
    max_points : list of dicts
        Output from get_max_points function
    method : str
        Aggregation method: 'weighted_mean', 'top1', 'mean', 'median'

    Returns:
    --------
    dict : {(TF, Target): {'pseudotime': float, 'force': float, ...}}
    """
    df = pd.DataFrame(max_points)

    aggregated = {}

    for (tf, target), group in df.groupby(['TF', 'Target']):
        if method == 'weighted_mean':
            # Normalize softmax weights within the group
            weights = group['softmax_weight'] / group['softmax_weight'].sum()
            agg_pseudotime = np.average(group['pseudotime'], weights=weights)
            agg_force = np.average(group['force'], weights=weights)
            agg_window_idx = int(np.round(np.average(group['window_idx'], weights=weights)))

        elif method == 'top1':
            # Take the one with highest softmax weight
            top_row = group.loc[group['softmax_weight'].idxmax()]
            agg_pseudotime = top_row['pseudotime']
            agg_force = top_row['force']
            agg_window_idx = top_row['window_idx']

        elif method == 'mean':
            agg_pseudotime = group['pseudotime'].mean()
            agg_force = group['force'].mean()
            agg_window_idx = int(np.round(group['window_idx'].mean()))

        elif method == 'median':
            agg_pseudotime = group['pseudotime'].median()
            agg_force = group['force'].median()
            agg_window_idx = int(np.round(group['window_idx'].median()))

        aggregated[(tf, target)] = {
            'pseudotime': float(agg_pseudotime),
            'window_idx': int(agg_window_idx),
            'force': float(agg_force),
            'abs_force': float(abs(agg_force)),
            'n_points': len(group)
        }

    return aggregated


def order_links_by_phase(regulation_pseudotimes):
    """
    Order TF-Target links into phases by ascending peak pseudotime.

    Parameters:
    -----------
    regulation_pseudotimes : dict
        Output from aggregate_max_points: {(TF, Target): {'pseudotime': float, ...}}

    Returns:
    --------
    list of (TF, Target) tuples sorted by their peak pseudotime (earliest first)
    """
    return sorted(
        regulation_pseudotimes,
        key=lambda link: regulation_pseudotimes[link]['pseudotime']
    )


def order_links(force_curves, dtime, top_k=5, temperature=1.0, method='weighted_mean'):
    """
    Convenience wrapper: compute the phase ordering of links directly from force curves.

    Parameters
    ----------
    force_curves : DataFrame
        Multi-indexed (TF, Target) force curves over pseudotime.
    dtime : array-like
        Pseudotime values per column/window.
    top_k, temperature :
        Softmax parameters passed to :func:`get_max_points`.
    method :
        Aggregation method passed to :func:`aggregate_max_points`.

    Returns
    -------
    tuple
        ``(ordered_links, regulation_pseudotimes)``:

        * ``ordered_links`` -- list of ``(TF, Target)`` tuples sorted by peak pseudotime.
        * ``regulation_pseudotimes`` -- dict from :func:`aggregate_max_points`.
    """
    max_points = get_max_points(force_curves, dtime, top_k=top_k, temperature=temperature)
    regulation_pseudotimes = aggregate_max_points(max_points, method=method)
    ordered_links = order_links_by_phase(regulation_pseudotimes)
    return ordered_links, regulation_pseudotimes


class RegulatoryPhases:
    """Split a lineage trajectory into temporal *phases* (phase-splitting base class).

    A *phase* is an interval of pseudotime delimited by cell-state termination
    pseudotimes (the "switches", from :class:`StateFrequency`). ``N`` switches ->
    ``N + 1`` phases (1-indexed): a pseudotime ``p`` lands in phase ``k`` where
    ``switch[k-2] < p <= switch[k-1]`` (the ``np.digitize(..., right=True)`` rule).
    So PB uses ``[ActB-4_termination, earlyPB_termination]`` (3 phases) and GC uses
    ``[ActB-3_termination]`` (2 phases).

    This base owns only the phase boundaries and the binning rule; subclasses add
    *what* is binned:

    - :class:`ForceWavePhases` -- classify TF-Target links by the softmax-weighted
      peak of their force wave.
    - :class:`BindingPhases` -- rank TF binding scores within each phase and pick
      the top-k TFs per category.

    The ``dynamic_validation`` TF-force validators reuse the static
    :meth:`assign_phases` (the force-wave binning) directly.
    """

    def __init__(self, switch_pseudotimes):
        """
        Parameters
        ----------
        switch_pseudotimes : sequence of float
            Cell-state termination pseudotimes separating consecutive phases
            (``N`` switches -> ``N + 1`` phases). Sorted on construction.
        """
        self.boundaries = np.sort(np.asarray(switch_pseudotimes, dtype=float))

    @property
    def n_phases(self):
        """Number of phases (``len(boundaries) + 1``)."""
        return len(self.boundaries) + 1

    def phase_of(self, pseudotime):
        """1-indexed phase of pseudotime value(s) (scalar or array)."""
        return np.digitize(pseudotime, self.boundaries, right=True) + 1

    @classmethod
    def from_states(cls, state_frequency, window_indices, boundary_states,
                    termination_method='threshold', threshold_frac=0.1,
                    prominence=10, distance=3, **kwargs):
        """Build from cell-state termination pseudotimes (the phase boundaries).

        Computes the termination pseudotime of each state in ``boundary_states``
        (in order) from a :class:`StateFrequency` instance, then constructs the
        phase splitter. Extra ``kwargs`` are forwarded to the subclass constructor
        (e.g. ``waves=`` for :class:`ForceWavePhases`; ``chromatin_object=`` and
        ``lineage=`` for :class:`BindingPhases`).

        Example
        -------
        PB (3 phases)::

            ForceWavePhases.from_states(sf_pb, PB_post_bifurcation_window_indices,
                                        ['ActB-4', 'earlyPB'], waves=waves_pb)
        """
        switch_pseudotimes = [
            state_frequency.termination_pseudotime(
                state, window_indices,
                method=termination_method, threshold_frac=threshold_frac,
                prominence=prominence, distance=distance,
            )
            for state in boundary_states
        ]
        return cls(switch_pseudotimes, **kwargs)

    @staticmethod
    def assign_phases(force_curves, dtime, switch_pseudotimes,
                      top_k=5, temperature=1.0, method='weighted_mean'):
        """Bin every link in ``force_curves`` to a phase by its softmax peak.

        ``switch_pseudotimes`` are the cell-state termination pseudotimes that
        separate consecutive phases; ``N`` switches -> ``N+1`` phases (1-indexed).
        Returns ``{(TF, Target): phase}``.
        """
        boundaries = np.sort(np.asarray(switch_pseudotimes, dtype=float))
        reg_pt = aggregate_max_points(
            get_max_points(force_curves, dtime, top_k=top_k, temperature=temperature),
            method=method,
        )
        return {
            link: int(np.digitize(info['pseudotime'], boundaries, right=True)) + 1
            for link, info in reg_pt.items()
        }


class ForceWavePhases(RegulatoryPhases):
    """Classify TF-Target links into phases by the softmax peak of their force wave.

    Each link's peak pseudotime is the softmax-weighted peak of its force wave
    (from ``waves.force_curves``); binning those peaks against the phase boundaries
    (inherited from :class:`RegulatoryPhases`) assigns a phase.
    """

    def __init__(self, switch_pseudotimes, waves, top_k=5, temperature=1.0,
                 method='weighted_mean'):
        """
        Parameters
        ----------
        switch_pseudotimes : sequence of float
            Phase boundaries (see :class:`RegulatoryPhases`).
        waves : TFForceWaves
            Fitted branch whose ``compute_forces(links)`` has been called so
            ``force_curves`` / ``dtime`` hold the force waves to classify.
        top_k, temperature, method :
            Softmax peak-pseudotime parameters (passed to the module helpers).
        """
        super().__init__(switch_pseudotimes)
        self.waves = waves
        self.top_k = top_k
        self.temperature = temperature
        self.method = method

    def link_peak_pseudotimes(self, links=None):
        """Softmax peak pseudotime per link.

        Returns the ``aggregate_max_points`` dict
        ``{(TF, Target): {'pseudotime': float, ...}}`` for ``links``
        (default: every link in ``waves.force_curves``).
        """
        if self.waves.force_curves is None:
            raise RuntimeError("Call waves.compute_forces(links) first.")
        df = (self.waves.force_curves if links is None
              else self.waves.force_curves.loc[links])
        max_points = get_max_points(df, self.waves.dtime,
                                    top_k=self.top_k, temperature=self.temperature)
        return aggregate_max_points(max_points, method=self.method)

    def classify_phases(self, links=None):
        """Assign each link to a phase by its softmax peak pseudotime.

        Each link's peak pseudotime is binned against the phase boundaries
        (``self.boundaries``) via :meth:`RegulatoryPhases.phase_of`. Build the
        instance from cell-state terminations with :meth:`RegulatoryPhases.from_states`
        (``ForceWavePhases.from_states(sf, window_indices, boundary_states, waves=...)``).

        Returns
        -------
        pandas.DataFrame with columns ``TF, Target, peak_pseudotime, phase``,
        sorted by phase then peak pseudotime.
        """
        reg_pt = self.link_peak_pseudotimes(links=links)

        rows = []
        for (tf, target), info in reg_pt.items():
            peak = info['pseudotime']
            phase = int(self.phase_of(peak))
            rows.append({'TF': tf, 'Target': target,
                         'peak_pseudotime': peak, 'phase': phase})

        return (
            pd.DataFrame(rows)
            .sort_values(['phase', 'peak_pseudotime'])
            .reset_index(drop=True)
        )


class BindingPhases(RegulatoryPhases):
    """Rank TF binding scores within each phase and pick the top-k TFs per category.

    Inherits the phase boundaries / binning from :class:`RegulatoryPhases` and binds
    a chromatin object (:class:`~firefate.temporal._chromatin.SmoothedCurvesChromatin`
    or any object exposing ``pb_pseudotime``/``gc_pseudotime`` and
    ``series_pb``/``series_gc``) on one lineage. A TF's *per-phase binding score* is
    the ``max`` of its smoothed binding-score curve over the windows falling in that
    phase -- the analogue of the abs-max TF force used for link validation, with the
    abs dropped since binding scores are non-negative.

    The selector replaces hand-picked TF panels: given a ``{category: [TF, ...]}``
    universe (e.g. the state-specific TFs vs the episodic TFs), it returns the top-k
    TFs **per phase** per category via :meth:`categories_by_phase`, as
    ``{phase: {category: {TF: color}}}``. Phases are lineage-specific differentiation
    windows (PB has 3, GC has 2), so picks are never pooled across phases. The
    per-phase selection drives the **box** view (:meth:`plot_box` / :meth:`top_tfs_table`);
    continuous binding-score / OCR curves over pseudotime are a separate concern,
    handled by ``SmoothedCurvesChromatin`` in the ``pseudotime_curves`` module.
    """

    #: default per-category sequential colormaps (cycled in category order)
    _DEFAULT_CMAPS = ('Purples', 'Oranges', 'Greens', 'Blues', 'Reds')

    def __init__(self, switch_pseudotimes, chromatin_object, lineage,
                 window_indices=None):
        """
        Parameters
        ----------
        switch_pseudotimes : sequence of float
            Phase boundaries (cell-state termination pseudotimes) for this lineage;
            ``N`` switches -> ``N + 1`` phases. Use :meth:`RegulatoryPhases.from_states`
            to derive them from a :class:`StateFrequency`.
        chromatin_object : SmoothedCurvesChromatin-like
            ``process_dynamics`` must already have run so ``series_pb``/``series_gc``
            (TF -> smoothed binding-score curve over this lineage's windows) and
            ``pb_pseudotime``/``gc_pseudotime`` are populated.
        lineage : {'pb', 'gc'}
            Which lineage's binding series / pseudotimes to bin.
        window_indices : sequence of int, optional
            Window IDs to restrict the binning to -- pass this lineage's
            post-bifurcation windows (the same list given to :class:`StateFrequency`,
            e.g. ``PB_post_bifurcation_window_indices``). The chromatin object spans
            the *full* trajectory, including the shared pre-bifurcation trunk whose
            windows (and binding-score values) are identical for PB and GC; binning
            that trunk collapses both lineages onto the same top TFs. Restricting to
            the post-bifurcation windows keeps each lineage's phases lineage-specific.
            When ``None`` the full lineage series is used (legacy behaviour).
        """
        super().__init__(switch_pseudotimes)
        self.lineage = lineage
        if lineage == 'pb':
            pseudotime = np.asarray(chromatin_object.pb_pseudotime)
            series = chromatin_object.series_pb
            traj_windows = np.asarray(chromatin_object.pb_indices)
        elif lineage == 'gc':
            pseudotime = np.asarray(chromatin_object.gc_pseudotime)
            series = chromatin_object.series_gc
            traj_windows = np.asarray(chromatin_object.gc_indices)
        else:
            raise ValueError("lineage must be 'pb' or 'gc'.")
        if not series:
            raise ValueError(
                "chromatin_object has no processed series; call process_dynamics() first.")
        if window_indices is not None:
            keep = np.isin(traj_windows, np.asarray(list(window_indices)))
            pseudotime = pseudotime[keep]
            series = {tf: np.asarray(v)[keep] for tf, v in series.items()}
        self.pseudotime = pseudotime
        self.series = series
        # phase index (1-indexed) of every window on this lineage
        self.window_phase = self.phase_of(self.pseudotime)
        empty = [ph for ph in range(1, self.n_phases + 1)
                 if not np.any(self.window_phase == ph)]
        if empty:
            warnings.warn(
                f"BindingPhases[{lineage}]: phase(s) {empty} have no windows -- the "
                f"switch pseudotimes {list(np.round(self.boundaries, 4))} fall outside "
                f"the binned pseudotime range "
                f"[{self.pseudotime.min():.4f}, {self.pseudotime.max():.4f}]. "
                "Check that the switches and the chromatin window pseudotimes share the "
                "same AlignTimeScales frame.",
                stacklevel=2,
            )

    @staticmethod
    def tfs_from_links(links):
        """Unique TF names from an iterable of ``(TF, Target)`` links (order-preserving)."""
        return list(dict.fromkeys(tf for tf, _ in links))

    @staticmethod
    def disjoint_categories(category_tfs):
        """Make the category TF universes mutually exclusive (union minus intersection).

        A TF that appears in more than one category (e.g. CREB3L2 / TFEC in both the
        state-specific and episodic sets) is dropped from *every* category, so each
        category is sampled only from the TFs unique to it. Order within each category
        is preserved.
        """
        counts = {}
        for tfs in category_tfs.values():
            for tf in dict.fromkeys(tfs):
                counts[tf] = counts.get(tf, 0) + 1
        return {cat: [tf for tf in dict.fromkeys(tfs) if counts[tf] == 1]
                for cat, tfs in category_tfs.items()}

    def phase_score(self, tf, phase):
        """Max binding score of ``tf`` over the windows in ``phase`` (NaN if none)."""
        if tf not in self.series:
            return np.nan
        vals = np.asarray(self.series[tf], dtype=float)[self.window_phase == phase]
        vals = vals[~np.isnan(vals)]
        return float(vals.max()) if vals.size else np.nan

    def rank_tfs(self, tfs, phase):
        """``tfs`` ranked as ``(TF, score)`` by descending per-phase binding score.

        TFs absent from the series, or with no windows in the phase (NaN score), are
        dropped.
        """
        scored = [(tf, self.phase_score(tf, phase)) for tf in dict.fromkeys(tfs)]
        scored = [(tf, s) for tf, s in scored if not np.isnan(s)]
        return sorted(scored, key=lambda kv: kv[1], reverse=True)

    def top_tfs(self, tfs, phase, top_k=5):
        """Names of the top-``top_k`` TFs in ``phase`` by binding score."""
        return [tf for tf, _ in self.rank_tfs(tfs, phase)[:top_k]]

    def phase_pool(self, phase, exclude=(), min_score=0.0):
        """All eligible ``(TF, score)`` controls in ``phase``, sorted by score.

        Every TF with a non-NaN per-phase binding score at or above ``min_score`` and
        not in ``exclude`` (typically the named-category TFs). With the default
        ``min_score=0.0`` this is every measured TF, including non-binders (score 0) --
        matching the 0-to-max population the named categories are drawn from. This is
        the full pool the size-matched random control samples from.
        """
        exclude = set(exclude)
        pool = [(tf, self.phase_score(tf, phase)) for tf in self.series
                if tf not in exclude]
        pool = [(tf, s) for tf, s in pool if not np.isnan(s) and s >= min_score]
        return sorted(pool, key=lambda kv: kv[1], reverse=True)

    def random_tfs(self, phase, k=5, exclude=(), rng=None, min_score=0.0):
        """``k`` random ``(TF, score)`` controls drawn from TFs scored in ``phase``.

        Size-matched random control for one phase: samples (without replacement) from
        :meth:`phase_pool` (every TF with a non-NaN per-phase binding score at or above
        ``min_score`` and not in ``exclude``, so the control never overlaps the named
        categories). Returns fewer than ``k`` only when the eligible pool is smaller,
        sorted by descending score for display. Pass a seeded ``numpy.random.Generator``
        as ``rng`` (and reuse it across phases) for reproducible draws.
        """
        rng = np.random.default_rng() if rng is None else rng
        pool = self.phase_pool(phase, exclude=exclude, min_score=min_score)
        if len(pool) > k:
            chosen = rng.choice(len(pool), size=k, replace=False)
            pool = [pool[int(i)] for i in chosen]
        return sorted(pool, key=lambda kv: kv[1], reverse=True)

    def top_tfs_table(self, category_tfs, top_k=5, exclusive=False,
                      random_control=False, control_name='Random', random_state=None):
        """Long table of the top-``top_k`` TFs per (phase, category).

        ``category_tfs`` maps each category name to its TF universe (an iterable of
        TF names; pass :meth:`tfs_from_links` output for link sets). When
        ``exclusive=True`` the categories are first made mutually exclusive via
        :meth:`disjoint_categories` (TFs shared across categories are dropped from
        all of them). When ``random_control=True`` an extra ``control_name`` category
        is added per phase: ``top_k`` random TFs (size-matched), drawn via
        :meth:`random_tfs` from TFs scored in that phase but absent from every named
        category, for a binding-strength baseline. ``random_state`` seeds the draw.
        Returns a DataFrame with columns ``phase, category, rank, TF, binding_score``.
        """
        exclude = {tf for tfs in category_tfs.values() for tf in tfs}
        if exclusive:
            category_tfs = self.disjoint_categories(category_tfs)
        rng = np.random.default_rng(random_state)
        rows = []
        for phase in range(1, self.n_phases + 1):
            for cat, tfs in category_tfs.items():
                for rank, (tf, score) in enumerate(self.rank_tfs(tfs, phase)[:top_k], 1):
                    rows.append({'phase': phase, 'category': cat, 'rank': rank,
                                 'TF': tf, 'binding_score': score})
            if random_control:
                for rank, (tf, score) in enumerate(
                        self.random_tfs(phase, k=top_k, exclude=exclude, rng=rng), 1):
                    rows.append({'phase': phase, 'category': control_name, 'rank': rank,
                                 'TF': tf, 'binding_score': score})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # categories-style dicts (drop-in for the chromatin plotters)
    # ------------------------------------------------------------------

    def _resolve_cmaps(self, category_tfs, category_cmaps):
        if category_cmaps is not None:
            return category_cmaps
        return {cat: self._DEFAULT_CMAPS[i % len(self._DEFAULT_CMAPS)]
                for i, cat in enumerate(category_tfs)}

    @staticmethod
    def _shade(tfs, cmap):
        """``{TF: hex}`` colouring ``tfs`` darkest->lightest from ``cmap`` (rank order)."""
        if not tfs:
            return {}
        cmap = plt.get_cmap(cmap)
        levels = np.linspace(0.85, 0.35, len(tfs))
        return {tf: to_hex(cmap(level)) for tf, level in zip(tfs, levels)}

    def categories_by_phase(self, category_tfs, top_k=5, category_cmaps=None,
                            exclusive=False):
        """``{phase: {category: {TF: color}}}`` of the top-``top_k`` TFs per (phase, category).

        Phases are this lineage's differentiation windows (PB: 3, GC: 2); each phase
        gets its own independent top-``top_k`` ranking -- picks are never pooled across
        phases. ``category_tfs`` maps each category to its TF universe. ``category_cmaps``
        optionally maps each category to a matplotlib colormap name (default: cycle
        :attr:`_DEFAULT_CMAPS`); within a category TFs are shaded darkest (highest
        binding score) to lightest. This drives the per-phase **box** view
        (:meth:`plot_box`); continuous binding/OCR curves over pseudotime are a
        separate concern handled by ``SmoothedCurvesChromatin`` in the
        ``pseudotime_curves`` module. When ``exclusive=True`` the categories are
        first made mutually exclusive via :meth:`disjoint_categories`.
        """
        if exclusive:
            category_tfs = self.disjoint_categories(category_tfs)
        cmaps = self._resolve_cmaps(category_tfs, category_cmaps)
        return {
            phase: {cat: self._shade(self.top_tfs(tfs, phase, top_k=top_k), cmaps[cat])
                    for cat, tfs in category_tfs.items()}
            for phase in range(1, self.n_phases + 1)
        }

    def plot_box(self, category_tfs, top_k=5, figsize=(8, 6),
                 ylabel='TF binding score (in-phase max)', category_cmaps=None,
                 annotate=False, exclusive=False,
                 random_control=False, control_name='Random', random_state=None,
                 control_cmap='Greys', violin=False, violin_width=1.6,
                 significance=False):
        """Per-phase box plot of the top-``top_k`` TFs' binding scores, grouped by category.

        One cluster of boxes per phase of this lineage; within each cluster one box
        per category, holding *that phase's* top-``top_k`` TFs' per-phase binding
        scores (the in-phase max -- the metric the selection ranks on). Each point is
        one selected TF; with ``annotate=True`` the TF names are drawn beside them.
        Phases are never pooled, so a TF appears only under the phase(s) it tops.

        Continuous binding-score / OCR curves over pseudotime are *not* drawn here --
        use ``SmoothedCurvesChromatin`` in the ``pseudotime_curves`` module for those.
        When ``exclusive=True`` the categories are first made mutually exclusive via
        :meth:`disjoint_categories` (TFs shared across categories are dropped). When
        ``random_control=True`` an extra ``control_name`` box (coloured ``control_cmap``)
        is drawn per phase from ``top_k`` size-matched random TFs scored in that phase
        (seeded by ``random_state``), as a binding-strength baseline.

        When ``violin=True`` a translucent violin of the top half (by score) of each
        *named* category's in-phase distribution is drawn behind its box; the box/
        scatter/annotations still show only the top-``top_k``.
        The random control stays box-only (its size-matched sample is the baseline; a
        violin of the whole pool behind it would be redundant). ``violin_width`` scales
        the violin width relative to the per-category slot.

        When ``significance=True`` and exactly two *named* categories are present
        (e.g. ``static`` and ``episodic``), a significance bar is drawn per phase
        between their boxes, annotated with the two-sided Mann-Whitney U result
        (stars / ``ns``) computed on the plotted top-``top_k`` TFs of each
        category in that phase.
        """
        table = self.top_tfs_table(category_tfs, top_k=top_k, exclusive=exclusive,
                                   random_control=random_control,
                                   control_name=control_name, random_state=random_state)
        phases = list(range(1, self.n_phases + 1))
        cats = list(category_tfs) + ([control_name] if random_control else [])
        cmaps = self._resolve_cmaps(category_tfs, category_cmaps)
        if random_control:
            cmaps = {**cmaps, control_name: control_cmap}
        colors = [to_hex(plt.get_cmap(cmaps[c])(0.6)) for c in cats]

        # top-half in-phase distribution behind each NAMED-category box (upper 50% of
        # the universe by score, not just top_k). The random control is box-only -- a
        # violin of the whole pool behind 5 arbitrary draws would just be redundant.
        full = {}
        if violin:
            vcats = self.disjoint_categories(category_tfs) if exclusive else category_tfs
            for p in phases:
                for cat, tfs in vcats.items():
                    scores = [s for _, s in self.rank_tfs(tfs, p)]  # descending
                    half = (len(scores) + 1) // 2  # top half (upper by score)
                    full[(p, cat)] = np.array(scores[:half], dtype=float)

        pair = None
        if significance:
            named = list(category_tfs)
            if 'static' in named and 'episodic' in named:
                pair = ('static', 'episodic')
            elif len(named) == 2:
                pair = (named[0], named[1])
            else:
                warnings.warn(
                    'significance=True needs exactly two named categories '
                    '(or both "static" and "episodic"); skipping significance bars.')

        return plot_phase_binding_boxes(
            table, phases, cats, colors, full=full, ylabel=ylabel, figsize=figsize,
            annotate=annotate, violin=violin, violin_width=violin_width,
            significance_pair=pair,
        )


def plot_force_heatmap_by_phase(
    force_curves,
    dtime,
    links,
    switch_pseudotimes,
    top_k=5,
    temperature=1.0,
    method='weighted_mean',
    cmap='RdBu_r',
    vmax=None,
    figsize=(4, 6),
    plot_figure=True,
    show_phase_dividers=True,
    ytick_fontsize=10,
):
    """Plot a force heatmap with links grouped into regulatory phases.

    Links are first binned into temporal phases by the softmax-weighted peak of
    their force wave (``switch_pseudotimes`` are the lineage-derived cell-state
    termination pseudotimes that separate consecutive phases -- the same
    boundaries used by :class:`RegulatoryPhases`; ``N`` switches -> ``N+1``
    phases). Within each phase rows are ordered by *ascending* peak pseudotime,
    so a link whose force wave peaks earlier sits higher and later-peaking links
    fall to lower rows. Both activations (positive force) and repressions
    (negative force) are shown on a symmetric diverging color scale.

    Parameters
    ----------
    force_curves : DataFrame
        Multi-indexed (TF, Target) force curves over pseudotime.
    dtime : array-like
        Pseudotime value per column/window.
    links : list of (TF, Target)
        Links to classify and plot.
    switch_pseudotimes : sequence of float
        Pseudotimes separating consecutive phases (e.g. cell-state termination
        pseudotimes from :class:`StateFrequency`).
    top_k, temperature, method :
        Softmax peak-pseudotime parameters (passed to the module helpers).
    cmap, vmax, figsize :
        Heatmap styling. ``vmax`` defaults to the data's max ``|force|`` and the
        scale is made symmetric (``vmin = -vmax``).
    show_phase_dividers : bool
        Draw a line between phase blocks and label each block.

    Returns
    -------
    tuple
        ``(ordered_df, phases, fig)``:

        * ``ordered_df`` -- DataFrame of force values, rows phase-then-peak ordered,
          indexed by ``"TF->Target"`` labels, columns the pseudotimes.
        * ``phases`` -- list of phase numbers aligned with ``ordered_df`` rows.
        * ``fig`` -- matplotlib Figure, or ``None`` when ``plot_figure`` is False.
    """
    existing_links = [link for link in links if link in force_curves.index]
    if not existing_links:
        raise KeyError("None of the requested links are in force_curves.index")
    sub = force_curves.loc[existing_links]
    phase_of = RegulatoryPhases.assign_phases(
        sub, dtime, switch_pseudotimes,
        top_k=top_k, temperature=temperature, method=method,
    )
    ordered, reg_pt = order_links(
        sub, dtime, top_k=top_k, temperature=temperature, method=method,
    )
    # Group by phase, then ascending peak pseudotime within each phase so the
    # earliest-peaking link is the top row of its block.
    ordered = sorted(ordered, key=lambda link: (phase_of[link], reg_pt[link]['pseudotime']))
    phases = [phase_of[link] for link in ordered]

    labels = [f"{tf}->{target}" for tf, target in ordered]
    dnet = np.array([sub.loc[link].values for link in ordered])
    ordered_df = pd.DataFrame(
        dnet, index=labels, columns=[f"{x:.4f}" for x in dtime]
    )

    fig = None
    if plot_figure:
        fig, _ = plot_phase_ordered_force_heatmap(
            dnet, labels, phases, dtime,
            cmap=cmap, vmax=vmax, figsize=figsize,
            show_phase_dividers=show_phase_dividers, ytick_fontsize=ytick_fontsize,
        )

    return ordered_df, phases, fig


# ---------------------------------------------------------------------------
# Figures: phase-resolved binding and force heatmaps
# ---------------------------------------------------------------------------

_PHASE_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}


def plot_phase_binding_boxes(
    table,
    phases,
    cats,
    colors,
    full=None,
    ylabel='TF binding score (in-phase max)',
    figsize=(8, 6),
    annotate=False,
    violin=False,
    violin_width=1.6,
    significance_pair=None,
):
    """Per-phase box plot of selected TFs' binding scores, grouped by category.

    ``table`` is the long ``phase, category, TF, binding_score`` table from
    ``BindingPhases.top_tfs_table``; ``phases`` and ``cats`` fix the axis order and
    ``colors`` gives one colour per category. ``full`` optionally maps
    ``(phase, category)`` to the in-phase score distribution drawn as a violin behind
    the box. When ``significance_pair`` names two categories, a Mann-Whitney U bar is
    drawn per phase between them.
    """
    full = {} if full is None else full

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n = len(cats)
    width = 0.8 / n
    for ci, cat in enumerate(cats):
        offset = (ci - (n - 1) / 2) * width
        positions, data, names = [], [], []
        for pi, p in enumerate(phases):
            sub = table[(table['phase'] == p) & (table['category'] == cat)]
            positions.append(pi + offset)
            data.append(sub['binding_score'].values)
            names.append(list(sub['TF']))
        if violin:
            vsets, vpos = [], []
            for pos, p in zip(positions, phases):
                d = full.get((p, cat), np.empty(0))
                if d.size > 1:
                    vsets.append(d)
                    vpos.append(pos)
            if vsets:
                parts = ax.violinplot(vsets, positions=vpos,
                                      widths=width * violin_width,
                                      showextrema=False)
                for body in parts['bodies']:
                    body.set_facecolor(colors[ci])
                    body.set_edgecolor('none')
                    body.set_alpha(0.25)
                    body.set_zorder(0)
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black'))
        for patch in bp['boxes']:
            patch.set_facecolor(colors[ci])
            patch.set_alpha(0.6)
        for pos, vals, nm in zip(positions, data, names):
            if len(vals) == 0:
                continue
            x = pos + (np.random.rand(len(vals)) - 0.5) * width * 0.5
            ax.scatter(x, vals, color=colors[ci], edgecolor='black',
                       linewidth=0.4, s=22, zorder=3)
            if annotate:
                for xi, yi, ni in zip(x, vals, nm):
                    ax.text(xi, yi, f' {ni}', fontsize=7, va='center')

    if significance_pair is not None:
        c1, c2 = significance_pair
        off1 = (cats.index(c1) - (n - 1) / 2) * width
        off2 = (cats.index(c2) - (n - 1) / 2) * width
        lo, hi = ax.get_ylim()
        rng = hi - lo
        bar_top = hi
        for pi, p in enumerate(phases):
            d1 = table[(table['phase'] == p)
                       & (table['category'] == c1)]['binding_score'].values
            d2 = table[(table['phase'] == p)
                       & (table['category'] == c2)]['binding_score'].values
            if d1.size < 1 or d2.size < 1:
                continue
            try:
                _, pval = mannwhitneyu(d1, d2, alternative='two-sided')
            except ValueError:
                continue
            if pval < 1e-4:
                label = '****'
            elif pval < 1e-3:
                label = '***'
            elif pval < 1e-2:
                label = '**'
            elif pval < 0.05:
                label = '*'
            else:
                label = 'ns'
            top = max(d1.max(), d2.max(),
                      full.get((p, c1), np.array([-np.inf])).max(),
                      full.get((p, c2), np.array([-np.inf])).max())
            h = top + 0.03 * rng
            tick = 0.015 * rng
            x1, x2 = pi + off1, pi + off2
            ax.plot([x1, x1, x2, x2], [h, h + tick, h + tick, h],
                    color='black', lw=1.0, clip_on=False)
            ax.text((x1 + x2) / 2, h + tick, label,
                    ha='center', va='bottom', fontsize=9)
            bar_top = max(bar_top, h + 2 * tick)
        ax.set_ylim(lo, bar_top + 0.04 * rng)

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([f'Phase {_PHASE_ROMAN.get(p, p)}' for p in phases])
    ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[ci], alpha=0.6)
               for ci in range(n)]
    ax.legend(handles, cats, frameon=False)
    return fig, ax


def plot_phase_ordered_force_heatmap(
    dnet,
    labels,
    phases,
    dtime,
    cmap='RdBu_r',
    vmax=None,
    figsize=(4, 6),
    show_phase_dividers=True,
    ytick_fontsize=10,
):
    """Force heatmap whose rows are already grouped by phase then peak pseudotime.

    ``dnet`` is the (links x windows) force matrix, ``labels`` the ``"TF->Target"``
    row labels and ``phases`` the phase number of each row (contiguous per block).
    The colour scale is symmetric so activation and repression read alike.
    """
    vmax_val = float(np.abs(dnet).max()) if vmax is None else vmax
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dnet, aspect='auto', interpolation='none', cmap=cmap,
                   vmin=-vmax_val, vmax=vmax_val)
    plt.colorbar(im, ax=ax, label="Force")

    ax.set_xlabel("Pseudotime")
    num_ticks = min(10, dnet.shape[1])
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    dtime_arr = np.asarray(dtime)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{dtime_arr[i]:.4f}" for i in tick_positions],
                       rotation=45, ha="right")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=ytick_fontsize)

    if show_phase_dividers:
        start = 0
        for phase in sorted(set(phases)):
            count = phases.count(phase)  # phases is contiguous per block
            if start > 0:
                ax.axhline(start - 0.5, color="black", linewidth=1.5)
            ax.text(0.2, start + 0.05, f"Phase {phase}",
                    va="top", ha="left", fontsize=ytick_fontsize,
                    fontweight="bold",
                    bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
            start += count

    plt.tight_layout()
    return fig, ax
