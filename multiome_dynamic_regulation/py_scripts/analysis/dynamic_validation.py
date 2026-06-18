from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TFForceValidation builds its force selector from TFForceWaves.ForceSelector and
# bins links to phases via RegulatoryPhases.assign_phases (importing temporal_clustering
# also runs its ensure() so the firefate package is on the path).
from methods.FIREFate.multiome_dynamic_regulation.py_scripts.analysis.state_dynamics import TFForceWaves, RegulatoryPhases

# ---------------------------------------------------------------------------
# Class: validation of enriched links vs random links
# ---------------------------------------------------------------------------

class TFForceValidation:
    """Validate enriched (FireFate) links against size-matched random links by
    abs-max TF force, pooled across the whole trajectory (no phase structure).

    The comparison metric is the absolute maximum TF force along the trajectory
    (``max_t |force(t)|``), one value per link. Forces are picked through a
    :class:`TFForceWaves.ForceSelector`, so the comparison can be run on a single
    lineage (``mode='lineage'``) or combined across lineages (``mode='combined'``):

    * ``'lineage'`` -- enriched and random links are both scored on one branch.
    * ``'combined'`` -- each enriched link is scored on its strongest lineage
      (``max`` across branches), while the random null is the *union* of the
      per-branch random forces (every random link contributes its single-branch
      force on each branch, not a cross-branch max). In this mode the result
      carries a ``branch`` column naming the winning lineage per enriched link.

    The nested :class:`PhaseSplitted` (via :meth:`split_by_phase`) reuses this
    class to split the comparison by branch-qualified phase.
    """

    def __init__(self, waves, enriched_links, varname='w_in', mode='lineage'):
        """
        Parameters
        ----------
        waves : TFForceWaves, dict {name: TFForceWaves}, or TFForceWaves.ForceSelector
            The fitted branch(es) supplying forces. A single branch (or dict with
            one entry) supports ``mode='lineage'``; pass multiple branches for
            ``mode='combined'``. ``compute_forces(enriched_links)`` should have
            been called on each branch so the enriched forces are cached (random
            links are scored on demand).
        enriched_links : list of (TF, Target)
            The enriched / prioritized links (e.g. ``PB_links_plotting``).
        varname : str
            Network variable used for random-link forces (matches the enriched
            ``compute_forces`` call). Default ``'w_in'``.
        mode : {'lineage', 'combined'}
            See the class docstring.
        """
        self.selector = (waves if isinstance(waves, TFForceWaves.ForceSelector)
                         else TFForceWaves.ForceSelector(waves, varname=varname))
        self.enriched_links = [tuple(l) for l in enriched_links]
        self.varname = varname
        self.mode = mode
        self.result_ = None      # tidy DataFrame after run()

    # ------------------------------------------------------------------
    # random-pool helpers (shared by all validators)
    # ------------------------------------------------------------------

    def _build_random_pool(self, n_tf, n_target, exclude, rng):
        """Sample a pool of non-enriched random links (``n_tf`` x ``n_target``).

        ``n_tf`` / ``n_target`` default (when ``None``) to the number of unique
        enriched TFs / targets, so the pool mirrors the shape of the enriched set.
        ``exclude`` selects the null (see :meth:`run`). Returns the list of
        ``(TF, Target)`` pool links.
        """
        d = self.selector.any_waves().dictys_dynamic_object
        nname = np.asarray(d.nname)
        tf_universe = nname[np.asarray(d.nids[0])]
        target_universe = nname[np.asarray(d.nids[1])]
        enriched_set = set(self.enriched_links)
        enr_tfs = {tf for tf, _ in self.enriched_links}
        enr_targets = {tg for _, tg in self.enriched_links}

        # default pool size = shape of the enriched set it is compared against
        if n_tf is None:
            n_tf = len(enr_tfs)
        if n_target is None:
            n_target = len(enr_targets)

        if exclude == 'tf_and_target':
            cand_tfs = np.array([t for t in tf_universe if t not in enr_tfs])
            cand_targets = np.array([g for g in target_universe if g not in enr_targets])
        elif exclude == 'links':
            cand_tfs = np.asarray(tf_universe)
            cand_targets = np.asarray(target_universe)
        else:
            raise ValueError("exclude must be 'tf_and_target' or 'links'.")

        pick_tfs = rng.choice(cand_tfs, size=min(n_tf, len(cand_tfs)), replace=False)
        pick_targets = rng.choice(cand_targets, size=min(n_target, len(cand_targets)),
                                  replace=False)
        # drop exact enriched pairs (a no-op under 'tf_and_target', essential under 'links')
        return [(tf, tg) for tf in pick_tfs for tg in pick_targets
                if (tf, tg) not in enriched_set]

    def _enriched_forces(self):
        """``(present_links, {link: abs_max_force}, {link: branch})`` for the
        enriched links.

        Forces are picked via the selector under ``self.mode`` (cached enriched
        forces reused per branch). ``branch`` is the winning lineage under
        ``mode='combined'`` and the single scored lineage under ``'lineage'``.
        """
        if self.mode == 'combined':
            combined = self.selector.combined_abs_max_force(self.enriched_links)
            enr_force = {l: d['abs_max_force'] for l, d in combined.items()}
            enr_branch = {l: d['branch'] for l, d in combined.items()}
        else:
            branch = self.selector.branches[0]
            enr_force = self.selector.abs_max_force(
                self.enriched_links, mode='lineage', branch=branch)
            enr_branch = {l: branch for l in enr_force}

        present = [l for l in self.enriched_links if l in enr_force]
        missing = [l for l in self.enriched_links if l not in enr_force]
        if missing:
            print(f"{type(self).__name__}: {len(missing)} enriched links absent from "
                  f"force curves, skipped: {missing}")
        return present, enr_force, enr_branch

    def _random_null_rows(self, pool_links):
        """Random-null rows for ``pool_links``.

        Under ``mode='combined'`` every random link contributes its single-branch
        force on each branch (the per-branch union), each tagged with the
        ``branch`` it was scored on; under ``'lineage'`` it contributes one force
        on the chosen branch (no ``branch`` column).
        """
        if self.mode == 'combined':
            return [{'link': l, 'group': 'random', 'branch': name, 'abs_max_force': f}
                    for name, am in self.selector.branch_abs_max_force(pool_links).items()
                    for l, f in am.items()]
        return [{'link': l, 'group': 'random', 'abs_max_force': f}
                for l, f in self.selector.abs_max_force(pool_links, mode='lineage').items()]

    # ------------------------------------------------------------------
    # present-edge random nulls (sample real edges, score force post hoc):
    # state-constrained (PB-2/GC-1 windows) and true (whole w_in graph)
    # ------------------------------------------------------------------

    def _qualifying_windows(self, cell_labels, states, threshold, cluster_column):
        """Window indices whose summed composition of ``states`` exceeds ``threshold``.

        Composition = (# cells of those states in the window) / (# cells assigned
        to the window), matching ``get_top_k_fraction_labels`` (all assigned cells
        in the denominator). ``cell_labels`` is the per-cell cluster table (or a
        path to a CSV); its row order must match the columns of the cell-window
        assignment matrix ``prop['sc']['w']``.
        """
        if isinstance(cell_labels, str):
            cell_labels = pd.read_csv(cell_labels, header=0)
        d = self.selector.any_waves().dictys_dynamic_object
        assign = (np.asarray(d.prop['sc']['w']) == 1).astype(float)  # (n_window, n_cell)
        in_states = np.isin(np.asarray(cell_labels[cluster_column]),
                            list(states)).astype(float)              # (n_cell,)
        n_total = assign.sum(axis=1)
        comp = np.divide(assign @ in_states, n_total,
                         out=np.zeros_like(n_total), where=n_total > 0)
        return np.where(comp > threshold)[0]

    def _sample_present_links(self, windows, varname, n, rng):
        """Sample ``n`` non-enriched links present in ``windows`` of the ``varname`` graph.

        A link is "present" if its ``prop['es'][varname]`` edge is nonzero in any
        of ``windows`` (OR'd across them); edge weight is otherwise irrelevant to
        sampling. Every enriched TF row and enriched target column is zeroed out
        first (the "not-of-nodes"). ``n`` links are then drawn without replacement
        from the surviving edges (sampled directly from the index arrays, so the
        full candidate list is never materialised). Returns a list of sampled
        ``(TF, Target)`` links.
        """
        d = self.selector.any_waves().dictys_dynamic_object
        net = d.prop['es'][varname]                 # (n_tf, n_target, n_window)
        present = np.zeros(net.shape[:2], dtype=bool)
        for win in windows:
            present |= (np.asarray(net[:, :, win]) != 0)

        nname = np.asarray(d.nname)
        tf_names = nname[np.asarray(d.nids[0])]
        target_names = nname[np.asarray(d.nids[1])]
        present[np.isin(tf_names, list({tf for tf, _ in self.enriched_links})), :] = False
        present[:, np.isin(target_names, list({tg for _, tg in self.enriched_links}))] = False

        ti, tj = np.where(present)
        if len(ti) < n:
            print(f"{type(self).__name__}: candidate pool has only {len(ti)} links "
                  f"(< {n} enriched); using all.")
        sel = (rng.choice(len(ti), size=min(n, len(ti)), replace=False)
               if len(ti) else [])
        return [(tf_names[ti[k]], target_names[tj[k]]) for k in sel]

    def _combined_enriched(self):
        """``(enr, present)``: cross-branch combined force per enriched link.

        ``enr`` maps each scorable enriched link to ``{'abs_max_force', 'branch'}``
        (the stronger lineage); ``present`` is the enriched links that were
        scorable, in order. Enriched forces are cached on the branches, so this is
        cheap to call.
        """
        enr = self.selector.combined_abs_max_force(self.enriched_links)
        present = [l for l in self.enriched_links if l in enr]
        return enr, present

    def _null_result(self, enr, present, sampled):
        """Tidy enriched-vs-random table for ``sampled`` random links.

        Both groups are scored by abs-max TF force across all branches (the
        stronger lineage per link, via :meth:`ForceSelector.combined_abs_max_force`)
        -- exactly how the enriched links were selected. ``enr``/``present`` come
        from :meth:`_combined_enriched`. Stores and returns the
        ``link, group, branch, abs_max_force`` DataFrame.
        """
        rnd = self.selector.combined_abs_max_force(sampled)

        rows = [{'link': l, 'group': 'enriched', 'branch': enr[l]['branch'],
                 'abs_max_force': enr[l]['abs_max_force']} for l in present]
        rows += [{'link': l, 'group': 'random', 'branch': rnd[l]['branch'],
                  'abs_max_force': rnd[l]['abs_max_force']} for l in sampled if l in rnd]
        self.result_ = pd.DataFrame(rows).sort_values('group').reset_index(drop=True)
        return self.result_

    # ------------------------------------------------------------------
    # public API (pooled, no phases)
    # ------------------------------------------------------------------

    def run(self, n_tf=None, n_target=None, exclude='tf_and_target', random_state=0):
        """Pooled enriched-vs-random table across the whole trajectory.

        A pool of ``n_tf`` x ``n_target`` non-enriched random links is scored; a
        size-matched subset (one random draw per enriched link) is kept as the
        null.

        Parameters
        ----------
        n_tf, n_target : int, optional
            Size of the random candidate pool. ``None`` (default) -> number of
            unique enriched TFs / targets, mirroring the enriched set's shape.
        exclude : {'tf_and_target', 'links'}
            How the random pool is kept "non-enriched":

            * ``'tf_and_target'`` (default): drop *every* enriched TF and *every*
              enriched target from the sampling universe.
            * ``'links'``: keep the full universe and only remove the exact
              enriched ``(TF, Target)`` pairs.
        random_state : int
            Seed for reproducible sampling.

        Returns
        -------
        pandas.DataFrame with columns ``link, group, abs_max_force`` (``group`` is
        ``'enriched'`` or ``'random'``). Under ``mode='combined'`` an extra
        ``branch`` column records the winning lineage for enriched rows and the
        lineage each random link was scored on. Also stored on ``self.result_``.
        """
        rng = np.random.default_rng(random_state)
        present, enr_force, enr_branch = self._enriched_forces()

        pool_links = self._build_random_pool(n_tf, n_target, exclude, rng)
        null_rows = self._random_null_rows(pool_links)

        n = len(present)
        if len(null_rows) < n:
            print(f"{type(self).__name__}: random null has only {len(null_rows)} "
                  f"links (< {n} enriched); using all. Increase n_tf/n_target.")
        idx = (rng.choice(len(null_rows), size=min(n, len(null_rows)), replace=False)
               if null_rows else [])

        if self.mode == 'combined':
            rows = [{'link': l, 'group': 'enriched', 'branch': enr_branch[l],
                     'abs_max_force': enr_force[l]} for l in present]
        else:
            rows = [{'link': l, 'group': 'enriched', 'abs_max_force': enr_force[l]}
                    for l in present]
        rows += [null_rows[i] for i in idx]

        self.result_ = (
            pd.DataFrame(rows).sort_values('group').reset_index(drop=True))
        return self.result_

    def plot(self, figsize=(5, 6), ylabel='Abs max TF force',
             colors=('#d1495b', '#9aa0a6')):
        """Pooled box plot of abs-max TF force, enriched vs random."""
        if self.result_ is None:
            self.run()
        return self._plot_pooled(self.result_, figsize, ylabel, colors)

    @staticmethod
    def _plot_pooled(df, figsize, ylabel, colors):
        """Two-box (enriched vs random) plot of ``abs_max_force`` from a tidy table."""
        groups = [('enriched', 'Enriched (FireFate)'),
                  ('random', 'Random (non-enriched)')]

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        data = [df[df['group'] == g]['abs_max_force'].values for g, _ in groups]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black'))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for pos, (vals, c) in enumerate(zip(data, colors)):
            if len(vals) == 0:
                continue
            x = pos + (np.random.rand(len(vals)) - 0.5) * 0.25
            ax.scatter(x, vals, color=c, edgecolor='black', linewidth=0.4, s=22, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([label for _, label in groups])
        ax.set_ylabel(ylabel)
        return fig, ax

    # ------------------------------------------------------------------
    # multi-set comparison: several enriched sets vs ONE shared random null
    # ------------------------------------------------------------------

    @classmethod
    def compare_sets(cls, branches, enriched_sets, varname='w_in',
                     exclude='tf_and_target', random_state=0):
        """Compare several enriched link sets against ONE shared random null.

        Each enriched set is scored by abs-max TF force across both branches (the
        stronger lineage per link, ``mode='combined'``). A single random null is
        drawn from the universe with every TF/target appearing in *any* set removed
        (the per-branch UNION pool, exactly as :meth:`run` builds it under
        ``mode='combined'``), then size-matched to the **largest** enriched set.

        Parameters
        ----------
        branches : TFForceWaves, dict {name: TFForceWaves}, or ForceSelector
            The fitted branch(es) supplying forces (e.g. ``{'PB': waves_pb,
            'GC': waves_gc}``).
        enriched_sets : dict {label: list of (TF, Target)}
            One entry per enriched box (e.g. ``{'State-specific': ss_links,
            'Episodic': ep_links}``). Insertion order is preserved.
        varname, exclude, random_state :
            As in :meth:`run` (``exclude`` controls how the null stays
            "non-enriched"; the seed is reproducible).

        Returns
        -------
        pandas.DataFrame with columns ``group, link, branch, abs_max_force`` where
        ``group`` is each set's label or ``'random'``. Plot it with
        :meth:`plot_multi`.
        """
        selector = (branches if isinstance(branches, TFForceWaves.ForceSelector)
                    else TFForceWaves.ForceSelector(branches, varname=varname))
        rng = np.random.default_rng(random_state)

        # a union validator drives the shared random pool (excludes every enriched
        # TF/target across all sets)
        all_links = [tuple(l) for s in enriched_sets.values() for l in s]
        v = cls(selector, enriched_links=all_links, varname=varname, mode='combined')

        rows, max_n = [], 0
        for label, links in enriched_sets.items():
            combined = selector.combined_abs_max_force([tuple(l) for l in links])
            present = [l for l in (tuple(x) for x in links) if l in combined]
            max_n = max(max_n, len(present))
            rows += [{'group': label, 'link': l, 'branch': combined[l]['branch'],
                      'abs_max_force': combined[l]['abs_max_force']} for l in present]

        null_rows = v._random_null_rows(v._build_random_pool(None, None, exclude, rng))
        if len(null_rows) < max_n:
            print(f"{cls.__name__}.compare_sets: random null has only "
                  f"{len(null_rows)} links (< {max_n}); using all.")
        idx = (rng.choice(len(null_rows), size=min(max_n, len(null_rows)), replace=False)
               if null_rows else [])
        rows += [null_rows[i] for i in idx]      # already tagged group='random'

        return pd.DataFrame(rows).reset_index(drop=True)

    @staticmethod
    def plot_multi(df, group_order, group_labels=None, figsize=(7, 6),
                   ylabel='abs(max TF-force)', colors=None):
        """One box per group in ``group_order`` from a :meth:`compare_sets` table.

        ``group_order`` lists the ``group`` values (set labels then ``'random'``)
        in plotting order; ``group_labels`` are the matching x-tick labels (default:
        ``group_order``). ``colors`` is one colour per group (default cycles a
        red/orange/grey palette).
        """
        if group_labels is None:
            group_labels = list(group_order)
        if colors is None:
            base = ['#d1495b', '#edae49', '#66a182', '#2e4057', '#9aa0a6']
            colors = [base[i % len(base)] for i in range(len(group_order))]

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        data = [df[df['group'] == g]['abs_max_force'].values for g in group_order]
        positions = list(range(len(group_order)))
        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black'))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for pos, (vals, c) in enumerate(zip(data, colors)):
            if len(vals) == 0:
                continue
            x = pos + (np.random.rand(len(vals)) - 0.5) * 0.25
            ax.scatter(x, vals, color=c, edgecolor='black', linewidth=0.4, s=22, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel(ylabel)
        return fig, ax

    # ------------------------------------------------------------------
    # per-phase view: factory + nested validator
    # ------------------------------------------------------------------

    def split_by_phase(self, switch_pseudotimes,
                       top_k=5, temperature=1.0, method='weighted_mean'):
        """Per-phase view of this validation (see :class:`PhaseSplitted`).

        Reuses this instance's selector, enriched links and random-pool machinery;
        only adds the phase binning. ``switch_pseudotimes`` is either one ordered
        sequence (lineage mode -- applied to the single branch) or a
        ``{branch: sequence}`` mapping (combined mode -- one set of phase
        boundaries per lineage). ``top_k, temperature, method`` are the softmax
        peak-pseudotime parameters, kept identical to the enriched assignment.
        """
        return TFForceValidation.PhaseSplitted(
            self, switch_pseudotimes,
            top_k=top_k, temperature=temperature, method=method)

    class PhaseSplitted:
        """Per-phase enriched-vs-random comparison, nested inside (and built from)
        a :class:`TFForceValidation` instance.

        TF force is now an established validation metric, so this only *adds* the
        phase split on top of the outer validator -- it reuses that instance's
        force selector, enriched links and random-pool sampling by composition
        (``self.v``) rather than duplicating them.

        Each enriched link is scored on its stronger lineage (``mode='combined'``)
        or the single branch (``mode='lineage'``) -- exactly the force the outer
        validator already picks -- and then binned into a *phase of that lineage*
        by the softmax peak-pseudotime rule (:meth:`RegulatoryPhases.assign_phases`)
        against that branch's cell-state termination pseudotimes. Phases are
        therefore branch-qualified: a link appears only under its winning branch's
        phase (e.g. PB 1-3, GC 1-2). Within each ``(branch, phase)`` cell a
        size-matched set of random non-enriched links -- drawn through the outer
        validator's :meth:`_build_random_pool`, scored and phase-binned on the same
        branch -- forms the null.
        """

        def __init__(self, validation, switch_pseudotimes,
                     top_k=5, temperature=1.0, method='weighted_mean'):
            """
            Parameters
            ----------
            validation : TFForceValidation
                The outer (already-built) validator supplying the selector,
                enriched links, mode and random-pool machinery.
            switch_pseudotimes : sequence or dict {branch: sequence}
                Phase boundaries. A single sequence is applied to the (single)
                lineage branch; a mapping gives one set of boundaries per lineage
                (combined mode). ``N`` boundaries -> ``N + 1`` phases.
            top_k, temperature, method :
                Softmax peak-pseudotime parameters, kept identical to the enriched
                phase assignment for a fair comparison.
            """
            self.v = validation
            self.switch_by_branch = self._normalize_switches(switch_pseudotimes)
            self.softmax_kwargs = dict(top_k=top_k, temperature=temperature, method=method)
            self.result_ = None

        def _normalize_switches(self, switch_pseudotimes):
            """``{branch: sorted boundaries}``: accepts a per-branch mapping or a
            single sequence (applied to the lineage's single branch)."""
            if isinstance(switch_pseudotimes, dict):
                return {b: np.sort(np.asarray(v, dtype=float))
                        for b, v in switch_pseudotimes.items()}
            branch = self.v.selector.branches[0]
            return {branch: np.sort(np.asarray(switch_pseudotimes, dtype=float))}

        def _assign_phases(self, force_curves, dtime, branch):
            """Bin every link in ``force_curves`` to a phase on ``branch`` (same
            softmax rule and boundaries used for the enriched links)."""
            return RegulatoryPhases.assign_phases(
                force_curves, dtime, self.switch_by_branch[branch], **self.softmax_kwargs)

        # ------------------------------------------------------------------
        # public API (per branch-qualified phase)
        # ------------------------------------------------------------------

        def run(self, n_tf=None, n_target=None, exclude='tf_and_target', random_state=0):
            """Build the per-(branch, phase) enriched-vs-random comparison table.

            Each enriched link is placed under its winning branch's phase; a pool
            of ``n_tf`` x ``n_target`` random non-enriched links (see
            :meth:`TFForceValidation._build_random_pool`) is scored and
            phase-binned on each branch, and within every ``(branch, phase)`` cell
            a random subset is drawn to match the enriched count there.

            Parameters
            ----------
            n_tf, n_target, exclude, random_state :
                See :meth:`TFForceValidation.run` (the same random-pool controls).

            Returns
            -------
            pandas.DataFrame with columns ``link, branch, phase, group,
            abs_max_force`` (``group`` is ``'enriched'`` or ``'random'``). Also
            stored on ``self.result_``.
            """
            rng = np.random.default_rng(random_state)
            present, enr_force, enr_branch = self.v._enriched_forces()

            # enriched links grouped by their (winning) lineage
            enr_by_branch = defaultdict(list)
            for link in present:
                enr_by_branch[enr_branch[link]].append(link)

            # one random pool, scored on each branch that carries enriched links
            # (union null -- a random link keeps its own single-branch force)
            pool_links = self.v._build_random_pool(n_tf, n_target, exclude, rng)

            rows = []
            n_per_cell = defaultdict(int)     # (branch, phase) -> # enriched links
            for branch, links in enr_by_branch.items():
                fc, dtime = self.v.selector.force_curves(links, branch=branch)
                phases = self._assign_phases(fc, dtime, branch)
                for link in links:
                    p = phases[link]
                    n_per_cell[(branch, p)] += 1
                    rows.append({'link': link, 'branch': branch, 'phase': p,
                                 'group': 'enriched', 'abs_max_force': enr_force[link]})

            for branch in enr_by_branch:
                fc, dtime = self.v.selector.force_curves(pool_links, branch=branch)
                rnd_phases = self._assign_phases(fc, dtime, branch)
                rnd_force = self.v.selector.abs_max(fc)
                pool_by_phase = defaultdict(list)
                for link, p in rnd_phases.items():
                    pool_by_phase[p].append(link)
                for (b, p), n in n_per_cell.items():
                    if b != branch:
                        continue
                    cands = pool_by_phase.get(p, [])
                    if len(cands) < n:
                        print(f"PhaseSplitted: {branch} phase {p} has only "
                              f"{len(cands)} random links in the pool (< {n} "
                              f"enriched); using all. Increase n_tf/n_target.")
                    chosen_idx = (rng.choice(len(cands), size=min(n, len(cands)),
                                             replace=False) if cands else [])
                    for i in chosen_idx:
                        link = cands[i]
                        rows.append({'link': link, 'branch': branch, 'phase': p,
                                     'group': 'random', 'abs_max_force': rnd_force[link]})

            self.result_ = (
                pd.DataFrame(rows)
                .sort_values(['branch', 'phase', 'group'])
                .reset_index(drop=True)
            )
            return self.result_

        def plot(self, figsize=(9, 6), ylabel='Abs max TF force',
                 colors=('#d1495b', '#9aa0a6')):
            """Box plot of abs-max TF force, enriched vs random, one group of boxes
            per branch-qualified phase (e.g. ``PB-1 PB-2 PB-3 GC-1 GC-2``)."""
            if self.result_ is None:
                self.run()
            df = self.result_
            cells = sorted(set(zip(df['branch'], df['phase'])))
            groups = [('enriched', 'Enriched (FireFate)'),
                      ('random', 'Random (non-enriched)')]
            width = 0.35

            fig, ax = plt.subplots(figsize=figsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for gi, (g, _label) in enumerate(groups):
                offset = (gi - 0.5) * width
                positions, data = [], []
                for ci, (branch, p) in enumerate(cells):
                    vals = df[(df['branch'] == branch) & (df['phase'] == p)
                              & (df['group'] == g)]['abs_max_force'].values
                    positions.append(ci + offset)
                    data.append(vals)
                bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color='black'))
                for patch in bp['boxes']:
                    patch.set_facecolor(colors[gi])
                    patch.set_alpha(0.6)
                # one jittered point per link
                for pos, vals in zip(positions, data):
                    if len(vals) == 0:
                        continue
                    x = pos + (np.random.rand(len(vals)) - 0.5) * width * 0.5
                    ax.scatter(x, vals, color=colors[gi], edgecolor='black',
                               linewidth=0.4, s=22, zorder=3)

            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([f'{b}-{p}' for b, p in cells])
            ax.set_ylabel(ylabel)
            handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[gi], alpha=0.6)
                       for gi in range(len(groups))]
            ax.legend(handles, [label for _, label in groups], frameon=False)
            return fig, ax
