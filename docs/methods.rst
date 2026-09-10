Methods
=======

**FocalFire** is a framework that focuses dense mechanistic models of gene regulation, through
interpretable machine learning, onto the components governing cell fate decisions. It offers
inference tasks that: (1) discover cellular programs (CPs) underlying contrasting cell states
using an interpretable machine learning model and characterize their phenotypic roles through
LLM-augmented gene set enrichment analysis; (2) construct state-specific gene regulatory networks
(GRNs) from single-cell multi-omics or single-cell RNA-seq data, together with a combined GRN whose
connectivity spans the cross-state scope of the cellular programs inferred in step 1; (3) construct
transition-window-specific GRNs from single-cell RNA-seq and single-cell ATAC-seq data with matched
or unmatched samples, and cluster their regulatory edges to temporally order waves of transcription
factor (TF) regulation; (4) construct episodic GRNs by retaining TF--target forces that remain
temporally invariant across an episode and fall in the top percentile of activating and repressive
edges; (5) quantify enrichment of dynamic TF activity within each discovered cellular program,
leveraging the mechanistic connectivity recovered by state-specific or episodic GRNs to offer
downstream links for the enriched TFs; (6) quantify an in silico perturbation score based on the
phenotypic shift caused by perturbing the enriched TFs in state-specific models; and (7) stratify
uncommitted state populations for fate bias using CPs inferred from intervention (gene or TF
knockout) versus unperturbed control populations.

We implemented and tested FocalFire in Python (versions xx) and designed it for use in a Jupyter
Notebook environment. FocalFire code is open source and available on GitHub at
`github.com/sachha-naksha/FocalFire <https://github.com/sachha-naksha/FocalFire>`_ and
`github.com/xxx <https://github.com/xxx>`_, along with detailed
function descriptions and tutorials at ``<READ-THE-DOCS>``. Additionally, we provide a
user-friendly web application at
`pitt-csi.shinyapps.io/firefate <https://pitt-csi.shinyapps.io/firefate/>`_ that facilitates
FocalFire analyses and enables
interactive exploration of the results presented in this manuscript.


Inferring cellular programs using an interpretable machine learning method
--------------------------------------------------------------------------

In the first step, the cellular programs underlying cell states are identified using scRNA-seq
data. To achieve this, we employ an interpretable machine learning method called SLIDE to identify
the significant cellular programs (CPs) underlying outcomes of interest from high-dimensional
omics datasets [ref to SLIDE]. This method is beneficial because it discovers necessary and
sufficient CPs that distinguish the states. SLIDE is a regression-based model that identifies
significant CPs, denoted by :math:`Z`, capturing linear and nonlinear relationships between
observed data :math:`X` and the phenotype of interest :math:`Y` (in our case, binary values
regarding each state of interest):

.. math::
   :label: eq-slide-factorization

   X' = A Z + E

Here, :math:`X \in \mathbb{R}^{n \times p}` is the scRNA-seq expression matrix with :math:`n`
cells and :math:`p` genes. :math:`X` is decomposed into two factors, :math:`A \in \mathbb{R}^{p
\times K}` and :math:`Z \in \mathbb{R}^{K \times n}`, with an error term :math:`E`. The matrix
:math:`A` is the allocation matrix and represents the membership of each feature to a cellular
program. The matrix :math:`Z` is the CP matrix and represents a lower-dimensional representation
(:math:`K < p`) of the input data in latent space. Using a regression model applied to CPs, the
linear part of the model is then generated to represent the significant standalone CPs:

.. math::
   :label: eq-slide-linear

   LP = \sum_{j \in S_1} \beta_j Z_j + \epsilon_1

Here, :math:`LP` is the linear part of the model, and :math:`S_1` is the set of significant
standalone CPs. Incorporating nonlinear relationships, the method then identifies significant
interacting CPs:

.. math::
   :label: eq-slide-nonlinear

   NP_j = \beta_j Z_j + \sum_i C_{ij} Z_i \odot Z_j + \epsilon_2,
   \quad
   i \neq j,\; j \in S_1,\; i \in \{1,\dots,K\},\; \text{and}\; Z_i \odot Z_j \in S_2

Here, :math:`\beta \in \mathbb{R}^{1 \times K}` and :math:`C_{ij}` is the effect size of the
interaction term between the two CPs :math:`Z_i` and :math:`Z_j`. :math:`S_2` is the set of
putative interactions involving the standalone significant CPs. These interactions arise from the
model's nonlinear architecture, capture combinatorial regulation, and improve interpretability and
predictive performance.

Before applying SLIDE, the single-cell RNA sequencing (scRNA-seq) data were preprocessed using the
filtering and preprocessing functions provided in the SLIDE library
(`github.com/jishnu-lab/SLIDE <https://github.com/jishnu-lab/SLIDE>`_) and the FocalFire GitHub repository
(`github.com/jishnu-lab/xxxx <https://github.com/jishnu-lab/xxxx>`_), following standard best practices. First, genes with
zero unique molecular identifier (UMI) counts across all cells were removed, and mitochondrial and
ribosomal genes were excluded from further analysis. Second, sparsity filtering was performed using
the ``zeroFiltering`` function in the SLIDE library to remove genes with more than ``col_thresh``
zero values across cells. To identify the cellular programs, SLIDE was applied to two distinct
scRNA-seq datasets to extract discriminatory regulons between predefined cell states in B cells
and T cells. SLIDE requires two input cell populations to learn latent regulatory programs that
differentiate one state from another. For the B-cell system, the method was applied to PB and GC B
cells using both scMultiome data (day 6) and scRNA-seq data. For the T-cell system, the model was
applied to Tex-term and Tex-KLR cells. A summary of the training conditions, including
experimental groups, collection time points, and model parameters, is provided in Supplementary
Table 1.

We also used this approach to derive predisposing cellular programs from single-cell Perturb-seq
data collected on day 6 following TF knockouts. The training dataset for B cells included knockout
conditions for key TFs implicated in B-cell differentiation (*PRDM1*, *BATF*, *SPIB*, *IRF4*, and
*IRF8*) alongside unperturbed control cells. For the T-cell system, we applied the same approach to
single-cell Perturb-seq data from *Ikzf1* and *Ets1* knockout cells against unperturbed control
cells. A summary of the training conditions, including experimental groups and model parameters, is
provided in Supplementary Table 2.


Interpreting cellular programs using LLM-based gene set enrichment analysis
---------------------------------------------------------------------------

To derive biologically meaningful modules and corresponding labels for each CP, we employed a
large language model (LLM)-based gene set enrichment approach on the top-ranked genes within each
CP. GSAI grouped genes into functional programs and generated module annotations. Using the
prompt provided in Supplementary File S1, we supplied the sets of top-ranked genes and asked
ChatGPT 5.1 to annotate them with the most relevant biological processes.


State-specific static GRN inference
-----------------------------------

GRNs were constructed separately for B-cell and T-cell datasets, each containing distinct
sub-cell types. For the B-cell dataset, GRNs were obtained for the PB and GC cells from [ref
Nick's]. For the T-cell dataset, GRNs were inferred for the Tex-term and Tex-KLR cells using the
CellOracle framework, with its murine ATAC-seq inferred GRN serving as the base GRN. For each cell
type (B or T), GRNs corresponding to the two relevant sub-cell types were first subset to the top
10,000 edges. These subtype-specific GRNs were then merged into a fusion GRN that represents the
combined regulatory landscape across sub-cell types within that dataset. During the merging
process, if an edge (TF--target interaction) was present in both subtype GRNs, the edge with the
higher weight, indicative of stronger regulatory potential, was retained. If an edge was unique to
one subtype, it was included as is in the fusion GRN. Following GRN fusion, each edge was
categorized based on its weight relative to the distribution of edge weights in the respective
fusion network: strong regulatory edges, defined as edge weights in the top tenth percentile (at
or above the 90th percentile), and weak regulatory edges, defined as edge weights below the 90th
percentile. This strategy enabled comprehensive subtype-integrated GRNs across functionally
distinct immune subpopulations.


Inferring regulons by combining cellular programs and GRNs
----------------------------------------------------------

To identify TFs with regulatory programs enriched in cell-subtype-discrimination-specific
regulons, we performed enrichment analyses using previously constructed fusion GRNs alongside
regulons inferred from SLIDE. By overlapping GRN topology with both standalone and interacting
regulons, we assembled a comprehensive set of TF--target gene relationships. These interactions
were then evaluated through two levels of enrichment analysis.

The first level was an order-1 (single-TF) enrichment analysis, in which, for each regulon, TFs
either directly included in the regulon or connected as first-degree neighbors based on GRN
topology were considered. To assess their regulatory importance, we quantified how many genes
each TF strongly regulates within the regulon compared with its global regulatory profile in the
fusion GRNs. For each TF, the enrichment score was computed as:

.. math::
   :label: eq-order1-enrichment

   \mathrm{Enrichment\ Score}
   =
   \frac{\dfrac{
     \text{Number of strong TF} \to \text{gene edges within the regulon}
   }{
     \text{Total number of genes in the regulon}
   }}{
   \dfrac{
     \text{Number of strong TF} \to \text{gene edges in the fusion GRNs}
   }{
     \text{Total number of genes in the GRN}
   }
   }

This enrichment score quantifies a TF's overrepresented strong regulatory influence for a given
regulon. A hypergeometric test was used to determine statistical significance, and transcription
factors with significantly enriched regulons were identified as candidate key regulators.

The second level was an order-2 (TF-pair) enrichment analysis. To capture potential combinatorial
regulation, we extended the analysis to transcription factor pairs. As in the order-1 analysis, we
evaluated TF pairs that were either members of a regulon or connected to regulon genes as
first-degree neighbors in fusion GRNs. For each TF pair, we identified the set of common
downstream genes that were co-regulated within the regulon. Only interactions where at least one
of the TF-to-gene edges was classified as strong were considered; weak--weak pairs were excluded.
The observed number of co-regulated genes within the regulon was then compared with the expected
overlap based on the global GRN:

.. math::
   :label: eq-order2-enrichment

   \mathrm{Enrichment\ Score}
   =
   \frac{\dfrac{
   \text{Number of } [(strong, strong), (strong, weak), (weak, strong)]
   \text{ co-regulated genes within the regulon}
   }{
     \text{Total number of genes in the regulon}
   }}{
   \dfrac{
   \text{Number of } [(strong, strong), (strong, weak), (weak, strong)]
   \text{ co-regulated genes in the fusion GRNs}
   }{
     \text{Total number of genes in the GRN}
   }
   }

To reduce spurious associations, we applied two filtering criteria: (1) TF pairs must
co-regulate more than one gene within the regulon, and (2) TF pairs must co-regulate at least four
genes across the full GRN. Statistical enrichment was again assessed using a hypergeometric test,
and TF pairs with significant enrichment (:math:`p < 0.05`) that passed both filters were
retained as candidate combinatorial regulators contributing to subtype-specific transcriptional
programs.


Benchmarking FocalFire cellular programs
----------------------------------------

To enable direct comparison of order-1 transcriptional programs enriched by FocalFire, namely
TF-centric regulons, we employed SCENIC+ to generate state-specific regulons distinguishing GC and
PB states in the B-cell system (Supplementary Figure 1), and Tex-term and Tex-KLR states in the
T-cell system (Supplementary Figure 2). Our workflow proceeded as follows. SCENIC+ was first applied
to single-cell multi-omic data from both systems to infer regulons differentially active between
the specified cell states. Regulon activity and specificity were quantified using the eRegulon
score. To ensure comparability across analyses, we retained only those SCENIC+-derived regulons
whose TFs overlapped with significantly enriched order-1 TFs identified through our regulon and
fusion GRN analyses. To compare the specificity and sensitivity of these TF-centric regulons, we
compiled the target genes of each transcription factor within the selected SCENIC+ regulons, as
well as those defined by the regulon and fusion GRN frameworks. We then calculated the absolute
fold changes in gene expression between the two cell-state subtypes to assess the differential
regulatory impact across methodologies.


Imputation of the multi-ome dataset
------------------------------------

PALANTIR was used to impute the scRNA counts of the B-cell multi-omic (scRNA+ATAC) dataset.
Following standard preprocessing, highly variable features were selected for PCA-based
dimensionality reduction. Ten principal components were then used to derive the diffusion maps
eigenvector basis and cellular neighborhood for the B-cell states. Palantir's built-in MAGIC
algorithm was used to impute the gene expression of the highly variable genes (:math:`n=3018`). The
imputed dataset was then filtered to retain only the cell states of interest for trajectory
inference, dropping the ``Naive``, ``8_UD_2``, and ``11_UD_3`` states.


Inferring pseudotime trajectories
---------------------------------

To reconstruct single-cell differentiation trajectories from our dataset of 28,494 cells in the
B-cell system, we applied STREAM to the MAGIC-imputed gene-expression layer using all highly
variable genes. A low-dimensional manifold of cellular neighborhoods reflective of the continuous
differentiation process was learned, and branching trajectories were inferred using STREAM's
standard locally linear embeddings (SE) and elastic principal graph framework, respectively. We
applied STREAM to 2,088 cells in the T-cell system from the non-targeted control population of the
RBPJ-knockout Perturb-seq dataset, without imputing gene expression. Raw gene-expression counts
were used to filter cells and features from the T-cell dataset, which were then scaled and
normalized. A total of 370 variable genes were selected for dimensionality reduction by fitting
a loess curve on the standard deviation and mean values of features with a fraction of 0.01 and
80th percentile. The parameters used for fitting the elastic graph on both the T-cell and B-cell
trajectories were ``epg_alpha = 0.02``, ``epg_mu = 0.07``, ``epg_lambda = 0.02``, and
``epg_trimmingradius = 2``. The method generated bifurcating and linear trajectories, capturing
the underlying transient cellular states in both the differentiating B cells and state-switching T
cells, and assigned pseudotime values to each cell to order them on a temporal scale. We used
STREAM's visualization modules to inspect trajectory branches, ordering of cell states based on
their sequential transitions, and gene-expression changes along pseudotime branches.


Inferring dynamic TF--target forces along pseudotime trajectories
-----------------------------------------------------------------

Smoothed regulatory curves along pseudotime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All dynamic analyses operate on a dynamic GRN reconstructed with dictys, in which a window of
cells is slid along the pseudotime trajectory and a context-specific network is inferred
independently within every window (194 windows for the B-cell system). FocalFire's temporal module
reads the resulting dynamic-network object and resolves it onto a continuous pseudotime axis
before any downstream quantity is computed.

For a branch delimited by a pair of trajectory endpoints, :math:`M` equidistant points are sampled
along that branch, and every per-window quantity :math:`q_w` is projected onto those points with a
Gaussian kernel of bandwidth :math:`d` in trajectory distance:

.. math::
   :label: eq-gaussian-smoothing

   \tilde{q}(t) = \frac{\sum_w K_d(t, w)\, q_w}{\sum_w K_d(t, w)},
   \qquad
   K_d(t, w) = \exp\left(-\frac{\delta(t, w)^2}{2 d^2}\right)

where :math:`\delta(t, w)` is the trajectory distance between sampled point :math:`t` and the
centroid of window :math:`w`. Smoothing is NaN-aware: windows in which a quantity is undefined
contribute no weight, and a sampled point is undefined only when every contributing window is.
Three families of curves are built this way.

*Expression curves* are the smoothed :math:`\log_2` CPM of every gene, :math:`x_g(t)`. The subset
restricted to transcription factors supplies the regulator abundance :math:`x_f(t)` used by the
force definition below. Target-gene responses are shown either as these curves directly or as
their pseudotime gradient :math:`\mathrm{d}x_g / \mathrm{d}t`, which makes the moment at which a
gene is induced or repressed, rather than its absolute level, the quantity compared across genes.

*Regulation curves* record how many genes a TF regulates at each point in pseudotime. At every
sampled point the smoothed network is binarized by retaining the
:math:`k = s \cdot n_{\mathrm{TF}} \cdot n_{\mathrm{target}}` strongest edges by absolute weight,
where :math:`s` is the network sparsity (:math:`s = 0.01` throughout), and each TF's out-degree
:math:`d_f(t)` in that binarized network is recorded as :math:`\log_2(d_f(t) + 1)`. A weighted
variant, in which the out-degree is summed over the weighted rather than the binarized network, is
available through the same interface. Because the per-point sparsity threshold and out-degree are
independent across sampled points, they are evaluated in parallel across cores.

*Beta curves* are the signed smoothed edge weights :math:`\beta_{f \to g}(t)` for a queried set of
TF--target links. Unless stated otherwise the normalized total-effect network (``w_in``) is used;
the direct-effect networks (``w_n`` and ``w``) are available through the same interface. Only the
queried sub-network is smoothed, and it is sliced before smoothing rather than after, so the full
TF :math:`\times` target :math:`\times` point tensor is never materialized.

Window pseudotimes and sampled-point pseudotimes are placed on a common scale
(``AlignTimeScales``) so that window-level quantities, such as cell-state composition and TF
binding, and point-level quantities, such as expression, :math:`\beta` and force, can be compared
on the same axis. For the B-cell system, episodic analyses used :math:`M = 40` sampled points with
:math:`d = 0.001` and force-wave analyses used :math:`M = 100` points with :math:`d = 0.0005`; the
T-cell system used :math:`M = 40` points with :math:`d = 0.002`.

Defining the dynamic TF force on a target gene
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fitted coefficient :math:`\beta_{i,j,t}` measures how strongly a TF :math:`i` can act on a
target gene :math:`j` at pseudotime :math:`t`, but not whether the factor is actually present;
conversely, TF expression alone says nothing about which targets a factor engages. The regulatory
force weights each TF--target edge by the transcriptional activity of the cognate TF, and is
defined as:

.. math::
   :label: eq-force-metric

   F(i,j,t) = \mathrm{sign}(\beta_{i,j,t}) \cdot \exp\left(\log_{10}\left(|\beta_{i,j,t}| + \epsilon\right) + \log_{10}\left(X_{i,t} + \epsilon\right)\right)

where :math:`\beta_{i,j,t}` is the regulatory coefficient for TF :math:`i` on target :math:`j` at
time :math:`t`, :math:`X_{i,t}` is the smoothed log-CPM expression of TF :math:`i` at time
:math:`t`, and :math:`\epsilon = 10^{-10}` is a small constant used to avoid numerical instability.
Neglecting :math:`\epsilon`, this is a monotone power transform,
:math:`\mathrm{sign}(\beta) \cdot (|\beta| \cdot X)^{1 / \ln 10}` with
:math:`1 / \ln 10 \approx 0.434`, of the product of regulatory strength and regulator abundance.
The force is therefore zero when either factor is absent, grows with both, inherits its sign from
:math:`\beta` so that positive force denotes activation and negative force repression, and the
logarithmic compression keeps edges whose coefficients span several orders of magnitude on a
comparable scale.

Evaluated over the sampled points of a branch, :math:`F(i,j,t)` forms the *force wave* of that
link. Two summaries of a force wave are used downstream: its mean over an episode, which ranks
edges when an episodic GRN is assembled, and its absolute maximum along a branch,
:math:`\max_t |F(i,j,t)|`, which is the per-link statistic used for phase assignment and for
validation.

Force waves are inspected in two complementary ways. For a single link, the force function is
drawn as a surface over regulator abundance and coefficient magnitude, one sheet for activation and
one for repression, and the link's own wave is traced on that surface, so that the path a link
takes through the abundance--coefficient plane and the force it generates are read off together.
For sets of links, the waves are stacked into a force heatmap over pseudotime on a symmetric
diverging scale, so that activating and repressing programs are directly comparable; rows and
columns can be ordered by hierarchical clustering of the waves themselves (Ward linkage on
Euclidean distance), which groups links that rise and fall together regardless of which TF drives
them.

Quantification of local TF activity across episodes of regulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A single GRN fitted over an entire transition trajectory averages over regulatory programs that
are active at different times. The trajectory branches for both datasets were therefore broken
into episodes of invariant TF activity. T cells have a linear trajectory divided into [N]
episodes, while B cells follow a bifurcating trajectory with [N] episodes per branch. Each episode
was defined as a contiguous slice of equidistantly sampled pseudotime windows from the trajectory.
Episodes are non-overlapping, and their boundaries were placed at approximately equidistant
intervals across the total number of sampled pseudotime points (:math:`n=40`), yielding
episode-specific subnetworks that capture temporally coherent regulatory states.

For each episode, a weighted regulatory episodic network was constructed by chunking the
smoothed network statistics (``stat.net``, ``stat.fbinarize``) at equidistantly sampled pseudotime
points within the episodic duration (:math:`n=5`). Regulatory edge weights (:math:`\beta`
coefficients) were retrieved for all TF--target pairs active within the episode. Edges that are
identically zero across the episode, that is, edges with no ATAC-seq support in the base GRN, were
dropped, and regulators whose gene symbols begin with ``ZNF`` or ``ZBTB`` were excluded, as these
large paralogous families lack the motif resolution to be assigned confidently. The surviving
edges were then filtered using two criteria applied in parallel: (i) a minimum number of non-zero
timepoint threshold (:math:`\geq 3`), ensuring the regulatory interaction was consistently
observed across the episode; and (ii) a one-sample :math:`t`-test against zero on the non-zero
:math:`\beta` values (:math:`\alpha = 0.05` at the filtering stage, tightened to
:math:`p < 0.001` for the episodic GRN), retaining only edges with statistically significant
regulatory signal. Additionally, direction invariance was enforced, requiring that all non-zero
:math:`\beta` values for a given TF--target pair share the same sign within the episode, thus
restricting retained edges to those exerting a consistent activating or repressing influence
throughout the episode.

For every retained edge the force :eq:`eq-force-metric` was computed at each sampled pseudotime
window of the episode from that edge's coefficients and the cognate TF's expression over the same
points. The average force across all timepoints within the episode was computed per edge, and the
top 2% of edges ranked by absolute average force were retained to form the episodic GRN. Where
activating and repressing programs were compared separately, the top 2% of positive and the bottom
0.5% of negative average forces were selected instead, so that repressive edges, which are far
fewer, are not lost to a single two-sided threshold. Episodes are reconstructed independently and
in parallel, one process per episode, so that no episode's filtering influences another's.

Enriching cellular programs against episodic GRNs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To identify TFs whose targets were enriched for genes of interest, here the genes of a cellular
program (a SLIDE latent factor, LF), a hypergeometric enrichment test was performed for each TF
within each episodic GRN. For a given TF with :math:`n` downstream targets in the episodic GRN, the
probability of observing :math:`k` or more LF-gene targets by chance was calculated as:

.. math::
   :label: eq-hg-episode

   P(X \geq k) = 1 - \mathrm{CDF}_{\mathrm{hypergeom}}(k - 1; N, K, n)

where :math:`N` is the total number of genes in the episodic GRN, :math:`K` is the total number of
LF genes active in the episode, and :math:`n` is the total number of targets regulated by the TF.
An enrichment score was computed as the fold enrichment over expectation,
:math:`k / (n \cdot K / N)`. TFs were ranked by enrichment score within each episode, and those
with zero enrichment were excluded. For each enriched TF the analysis also returns the program
genes it targets in that episode together with their average forces, and its remaining downstream
targets, so that the sign and strength of each regulatory relationship can be inspected alongside
the enrichment statistic. This framework enabled systematic, episode-resolved identification of
TFs whose downstream targets are disproportionately composed of fate-determining genes, providing
a temporally stratified view of regulatory control across the differentiation trajectory. Running
it across consecutive episodes yields the episodic enrichment pattern of a TF, the sequence of
episodes in which it acts on the program, which distinguishes factors that act throughout the
transition from those that act only within a restricted window of it.

Because episodes are reconstructed independently, an individual TF--target edge can also be
followed across them. For a chosen set of TFs and targets, each edge's average force is collected
from every episodic GRN into an edge :math:`\times` episode matrix, with zero entered wherever the
edge did not survive that episode's filtering; this matrix is what the TF--target episodic heatmaps
display. In the episodic enrichment dot plots, a TF is shown for an episode only if it targets at
least two program genes and at least two further downstream genes there. Unless an explicit order
is imposed, TFs are ordered by the episode in which they peak, namely the significant episode with
the highest enrichment score, falling back to all episodes when none is significant, and, within an
episode block, by decreasing peak enrichment score. Alternatively, TFs are ordered by the
similarity of the program genes they target, taking the Jaccard index between their target sets as
the similarity, :math:`1 -` Jaccard as the distance, and Ward linkage for the hierarchy. The same
TF--gene incidences, counted across episodes, give the TF--gene co-regulation map, which shows
which factors converge on the same program genes and in how many episodes they do so.

Analysis of global TF activity across trajectories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analysis of global TF activity across trajectories summarizes how a TF's activity is distributed
over an entire branch rather than within one episode. Each smoothed curve :math:`y(t)`, either
expression or regulation (:math:`\log_2` out-degree), sampled at pseudotimes
:math:`t_1 < \dots < t_M`, is reduced to four characteristics on a pseudotime axis rescaled to
:math:`[0, 1]`. Writing :math:`\mathrm{AUC}(y)` for the trapezoidal integral of :math:`y` over that
axis, and

.. math::
   :label: eq-endpoint-band

   \hat{y}(t) = \mathrm{median}\left(y(t),\, y(t_1),\, y(t_M)\right)

for the curve clipped to the band spanned by its endpoints, the characteristics are the terminal
log fold change :math:`\Delta_{\mathrm{term}} = y(t_M) - y(t_1)`, the transient log fold change
:math:`\Delta_{\mathrm{trans}} = \mathrm{AUC}(y - \hat{y})`, the switching time
:math:`\mathrm{AUC}(\hat{y} - \hat{y}(t_M)) / (\hat{y}(t_1) - \hat{y}(t_M))`, which locates the
pseudotime at which the main transition occurs, and :math:`\mathrm{AUC}(y)` itself. The terminal
term captures net change between the ends of the branch; the transient term captures excursions
away from the endpoint band and is by construction insensitive to net change.

Both terms are :math:`z`-scored across all curves, and each TF is assigned the wave pattern of
whichever term dominates in absolute value: *up* and *down* when
:math:`|z(\Delta_{\mathrm{term}})| \geq |z(\Delta_{\mathrm{trans}})|`, with the sign of
:math:`z(\Delta_{\mathrm{term}})` deciding between them, and *transiently up* (a bell-shaped wave)
and *transiently down* (a U-shaped wave) otherwise. TFs are additionally ranked within each pattern
by the absolute :math:`z`-score of the dominant term, which is how the top regulators per pattern
are selected for display. Transiently active regulators identified this way are exactly those that
a terminal-fold-change analysis, and any comparison of the two endpoint states, would miss.

Chromatin binding inference across pseudotime trajectories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TF binding was quantified directly from the per-window chromatin footprints produced during
dynamic-network inference. For each window, the inferred TF-to-open-chromatin-region (OCR) binding
table was read and, for every TF, two per-window quantities were computed: a *binding score*, the
mean footprint score of that TF's bound OCRs, averaged within chromosome and then across
chromosomes so that chromosomes with many peaks do not dominate; and a *binding count*, the mean
number of bound OCRs per chromosome computed the same way. Windows in which a TF has no bound
region contribute a missing score and a count of zero. TFs are either supplied explicitly or
discovered as the union of all factors observed across windows.

Each series is then ordered along a branch by that branch's window ordering, mapped onto the
branch's pseudotime axis via ``AlignTimeScales``, and smoothed with a one-dimensional Gaussian
filter (:math:`\sigma = 2` windows). Because the two branches of a bifurcating trajectory are
aligned separately, each branch's windows are mapped through its own pseudotime frame. When binding
is compared across branches, each TF's curve is optionally min--max scaled using the global minimum
and maximum across both branches together, so the two lineages remain on a shared reference scale.
These curves give an orthogonal, expression-independent readout of when a factor engages chromatin,
and are compared against the force waves of the same factors. The two binding metrics are also
compared against each other per TF, with the binding-score and OCR-count curves min--max scaled
onto a common axis, to distinguish a genuine strengthening of footprints from a simple increase in
the number of accessible regions bound.

Cell-state composition along the trajectory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dynamic network assigns each cell to one or more pseudotime windows. Combining this soft
cell-to-window assignment with the cell-state labels, we tabulate how many cells of each state fall
in every window, giving a state :math:`\times` window count table, and place each window on the
branch's pseudotime axis with ``AlignTimeScales``. The progenitor states shared by all branches
(``ActB-1`` and ``earlyActB`` in the B-cell system) are dropped, as they contribute to every
lineage alike and would mask lineage-specific structure. Restricting the table to the windows of
one branch gives that branch's composition trajectory, which is displayed both as per-state count
curves over pseudotime and as stacked bars of the average composition over eight equally spaced
bins of windows.

Each state's count trajectory is then reduced to its extrema: maxima are located by
prominence-based peak finding on the trajectory and minima by the same procedure on its negation
(prominence of 10 cells, minimum separation of 3 windows). A state's maximum marks its peak
abundance along the branch and the first minimum after it marks its collapse; the pseudotimes at
which states collapse are the cell-state switches that delimit the regulatory phases below.

Deriving phases of gene regulation governing state switching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Episodes divide a branch into equal blocks of pseudotime; *phases* instead divide it at the points
where the cellular composition of the trajectory changes, so that each phase corresponds to an
interval between two cell-state switches.

Phase boundaries are derived from the cell-state composition trajectories described above,
restricted to post-bifurcation windows so that the boundaries are lineage-specific. For a state
:math:`c`, its *termination pseudotime* is the first pseudotime after the state's peak abundance at
which its count falls to a fraction :math:`\tau` of that peak (:math:`\tau = 0.1`); alternatively
the first local minimum after the peak, located by peak finding on the negated count trajectory,
can be used, falling back to the threshold rule when no such minimum exists. If a state does not
collapse within the branch, the branch's final pseudotime is used, so the rule always returns a
usable boundary. In the B-cell system the PB branch is delimited by the terminations of ``ActB-4``
and ``earlyPB``, giving three phases, and the GC branch by the termination of ``ActB-3``, giving
two; in general :math:`N` switches define :math:`N + 1` phases, and a pseudotime :math:`p` falls in
phase :math:`k` when :math:`\mathrm{switch}_{k-1} < p \leq \mathrm{switch}_k`.

Each regulatory link is then assigned to the phase in which its force peaks. Rather than taking the
single largest point of a force wave, which is sensitive to the sampling of pseudotime, the peak is
estimated by softmax weighting of the wave's absolute values:

.. math::
   :label: eq-softmax-peak

   w_{i,j}(t) = \frac{\exp\left(|F(i,j,t)| / T\right)}{\sum_{t'} \exp\left(|F(i,j,t')| / T\right)}

with temperature :math:`T` controlling how peaked the weighting is. The :math:`\kappa`
highest-weighted points (:math:`\kappa = 5`) are retained and the link's peak pseudotime is their
weight-normalized mean, :math:`\hat{t}_{i,j} = \sum_t \tilde{w}(t)\, t`, where :math:`\tilde{w}`
renormalizes the softmax weights over the retained points. Taking the single top-weighted point, or
the unweighted mean or median of the retained points, is also supported. Binning
:math:`\hat{t}_{i,j}` against the phase boundaries assigns each link a phase, and ordering links by
:math:`\hat{t}` orders them by when their regulation acts. Because the phase is defined as the
interval containing the peak, :math:`\max_t |F(i,j,t)|` is simultaneously the link's in-phase peak
force.

The same phase partition is applied to the chromatin binding curves. A TF's *per-phase binding
score* is the maximum of its smoothed binding-score curve over the windows falling within that
phase, the binding analogue of the absolute maximum force, and TFs are ranked by that score
independently within each phase, so that selections are never pooled across phases. This provides
an unbiased, per-phase selection of the most strongly binding factors within a category, for
example state-specific versus episodic TFs, replacing hand-picked panels. Where two categories are
compared, TFs shared between them can be removed from both so that each category is sampled only
from the factors unique to it, a size-matched random control is drawn without replacement from the
remaining factors scored in that phase, and the two categories are compared within each phase with
a two-sided Mann--Whitney :math:`U` test.


Validating dynamic TF activity
-------------------------------

Comparison against random regulatory links
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To test whether the links prioritized by FocalFire carry more regulatory force than expected by
chance, each enriched link was compared against a size-matched null of non-enriched links using the
absolute maximum force :math:`\max_t |F(i,j,t)|` as the statistic, the same quantity by which links
were selected.

Null links were drawn in one of two ways. In the first, a candidate pool of
:math:`n_{\mathrm{TF}} \times n_{\mathrm{target}}` links is sampled from the network universe after
removing every enriched TF and every enriched target, or, in the more permissive variant, only the
exact enriched TF--target pairs; the pool is scored, and a random subset matched in size to the
enriched set is retained. In the second, links are sampled from edges that are actually present,
that is, non-zero in the fitted network, in windows whose cellular composition exceeds a threshold
for a given set of states, with all enriched TFs and targets masked out first; force is then scored
post hoc. The second null is the stricter of the two, since it holds fixed the fact that an edge
exists in the relevant cell states and asks only whether its force is comparable.

Because a bifurcating trajectory offers two lineages, each link can be scored either on one lineage
or across both. In the combined mode each enriched link is scored on the lineage on which its
absolute maximum force is larger, and that lineage is recorded alongside the value; the random null
is then the union of the per-branch forces of the sampled links, so the null is never given the
same cross-branch maximum advantage as the enriched set. Several enriched sets, for example
state-specific versus episodic links, can be compared against one shared null, drawn after removing
the TFs and targets of all sets and size-matched to the largest set.

The comparison is also carried out within phases. Each enriched link is assigned to a phase of its
winning lineage by the softmax peak rule :eq:`eq-softmax-peak`, random links are scored and
phase-binned on the same lineage with identical softmax parameters, and within every
(lineage, phase) cell the null is size-matched to the largest enriched set in that cell. This keeps
enriched and random links on the same footing at every step: the same network variable, the same
force definition, the same peak rule, and the same phase boundaries.

In-silico perturbation of FocalFire-prioritized TFs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The force-based comparison above asks whether the prioritized links are strong inside the dynamic
model that produced them. As an independent check we asked whether the factors FocalFire
prioritizes are also the ones whose loss most disturbs the fate decision in a model FocalFire did
not fit. Every transcription factor represented in the state-specific CellOracle GRN was knocked
out in silico, one factor at a time, and the resulting shift of each cell was scored against the
differentiation vector field of the trajectory, giving a per-cell perturbation score whose sign
records whether the knockout drives that cell with or against the differentiation flow.

Per-cell scores were then aggregated per lineage. Writing :math:`S^{+}_{\ell}` and
:math:`S^{-}_{\ell}` for the summed positive and negative perturbation scores over the cells of
lineage :math:`\ell`, the perturbation magnitude of that lineage is

.. math::
   :label: eq-perturbation-magnitude

   S_{\ell} = S^{+}_{\ell} + \left| S^{-}_{\ell} \right|

so that shifts with and against the differentiation flow both count as perturbation rather than
cancelling one another, and the *overall perturbation magnitude* of a factor is the sum over the
two terminal lineages, :math:`S = S_{\mathrm{PB}} + S_{\mathrm{GC}}` (2,789 PB and 6,342 GC cells
in the in vitro B-cell system). The lineage-resolved scores are retained alongside it, so that a
factor that perturbs one branch specifically can be distinguished from one that perturbs both.

Factors were then split into three groups: the state-specific TFs, enriched in the regulon and
fusion-GRN analysis; the episodic TFs, enriched in the episodic GRNs; and all remaining scored
factors, which form the background. Factors absent from the CellOracle GRN receive no perturbation
score and were excluded from every group, and the two factors enriched in both analyses (*CREB3L2*
and *TFEC*) were counted once when the two FocalFire groups were pooled. Each FocalFire group, and
their union, was compared against the background by a one-sided Mann--Whitney :math:`U` test on the
overall perturbation magnitude, testing the directional hypothesis that FocalFire-prioritized
factors carry larger perturbation magnitudes than the factors it did not prioritize.

Replication in an independent in vivo tonsil B-cell dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same in-silico perturbation analysis was repeated end to end in an independent human tonsil
B-cell dataset, in which the germinal-centre reaction proceeds in vivo rather than in culture and
the two terminal states are germinal-centre B cells (9,653 cells) and plasma cells (912 cells). GRN
inference, single-TF knockout simulation and perturbation scoring were carried out within that
dataset, so the vector field, the network and the scores are entirely independent of the in vitro
system. The FocalFire state-specific and episodic TF sets, restricted as before to the factors
present in the tonsil GRN, were then compared against the remaining scored factors with the same
one-sided Mann--Whitney :math:`U` test on the overall perturbation magnitude. Because the TF sets
were fixed before this dataset was scored, this is a direct test of whether the regulators FocalFire
identifies in vitro remain the high-impact regulators of the same fate decision in vivo.

Optimal-transport reconstruction of ancestor--descendant couplings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The force and perturbation analyses both operate inside a fitted GRN. To test the trajectory-level
claim independently of any network, namely that activated B cells are already biased toward a fate
before that fate is expressed, we reconstructed ancestor--descendant couplings of the in vitro
B-cell multi-ome directly from the data by optimal transport (OT), using the ``TemporalProblem`` of
moscot.

Cells were grouped into the three collection windows of the experiment (days 0--2, 3--4 and 5--6,
assigned numeric times 1.5, 3.5 and 5.5), and within each window only the sub-states actually
populated at that time were retained: ``earlyActB``, ``ActB-1`` and ``ActB-2`` at days 0--2;
``ActB-2``, ``ActB-3``, ``ActB-4`` and ``earlyPB`` at days 3--4; and ``GC-1``, ``GC-2`` and
``PB-2`` at days 5--6, each represented by more than 100 cells. RNA and ATAC measurements of the
same cells were embedded jointly with MultiVI, and the OT problems were solved in that shared
latent space.

Transport cost was taken from the geometry of the data rather than from Euclidean distance in the
embedding. For each consecutive pair of time points, a 30-nearest-neighbour connectivity graph was
built on the joint embedding of the two windows together and checked for connectedness, and that
graph was supplied as the cost of the linear term with diffusion time :math:`t = 100`, so that the
distance between two cells follows the data manifold instead of cutting across it. The entropically
regularized problems for the consecutive pairs :math:`1.5 \to 3.5` and :math:`3.5 \to 5.5` were
then solved with regularization :math:`\varepsilon = 10^{-3}` and mean cost scaling.

The resulting couplings give, for every pair of consecutive windows, the probability that a cell at
the earlier time gives rise to a cell at the later one. Aggregating a coupling by cell state gives
the forward (descendant) and backward (ancestor) transition matrices between the states of
consecutive windows; pushing a state forward or pulling a state backward through the couplings
gives, for every individual cell, its probability of descending from or giving rise to that state.
This is where the OT reconstruction and FocalFire meet: the pseudotime ordering, the episodes and
the phases assert an ancestor--descendant structure that the couplings estimate from the measured
data alone.

Driver transcription factors were then read off these probabilities in two complementary ways, in
both cases by correlating the normalized log-transformed expression of every annotated human TF
against a per-cell probability, with correlations reported alongside :math:`q`-values and a factor
called significant when :math:`q < 0.05` and :math:`|r| > 0.1`. In the *ancestor* direction, the
fate of interest was pulled back through both couplings, ``ActB-4`` pulled from days 3--4 to days
0--2 and ``GC-2`` pulled from days 5--6 to days 3--4, and the two per-cell pull weights were summed,
so that the resulting score marks the cells that lie on the path to that fate; TFs correlating
positively with it are expressed in the predisposed ancestors, and TFs correlating negatively mark
the cells that do not take that route. In the *descendant* direction, ``ActB-4`` was pushed forward
from days 3--4 to days 5--6, the source cells were masked so that only descendants remained, the
descendants were restricted to those annotated ``GC-2``, and TF expression was correlated against
the descendant probability, giving the factors that mark the phenotype the predisposed population
actually reaches. Recovering the same regulators here, from couplings estimated without any GRN,
pseudotime or force model, provides model-independent support for the dynamic TF activity that
FocalFire infers.


Fate prediction of early activated B cells
-------------------------------------------

To predict cell fate bias toward the final lineages, we first evaluated whether transcriptomic
features that distinguish TF-knockout cells from controls could also serve as informative
predictors. We identified key transcriptomic features that differentiate *PRDM1*- and
*IRF4*-perturbed cells from wild-type controls by training a lasso logistic regression
classification model (``LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000, random_state=42)``).
The trained model was then applied to early ABCs from the multi-omic dataset (days 2 and 4) using
the ``lasso.predict(...)`` function to generate predicted probabilities for each cell. Because the
response variable is binary, we assigned predicted fates to early ABCs based on whether their
predicted probability was above or below a threshold of 0.5.

The same procedure was repeated using only the differentially expressed genes from the training
datasets. We first performed differential expression analysis between knockout and control groups
and retained genes with an absolute :math:`\log_2` fold change greater than 1 and an adjusted
:math:`p`-value below 0.05. The single-cell gene-expression matrix was then subset to include only
these genes, and a lasso logistic regression model was trained on this reduced feature set. The
trained model was subsequently applied to early ABCs from the multi-omic dataset (days 2 and 4)
using the same prediction function to generate and project fate probabilities.

We then used the inferred cellular programs from single-cell TF Perturb-seq data in the B-cell
system. FocalFire projected these programs onto ABCs (days 2 and 4) from single-cell multi-ome
data to estimate their transcriptional predisposition toward one of two downstream fates (GC or
PB). The latent cellular programs defined a score for each cell, representing the degree to which
its expression profile aligned with either the knockout or control state. Using these scores, we
assessed the discriminative power of each model by examining how well the score distributions
separated knockout and control cell populations. For each model, we defined an optimal threshold
based on the distribution that most effectively distinguished these two groups. We then applied
the same trained models to the day 2 and day 4 ABC multi-ome data using the ``predZ`` function in
the SLIDE package, generating predicted scores for uncommitted cells. Using the thresholds defined
from the Perturb-seq models, FocalFire assigned a predicted fate score to each ABC. Cells with
predicted scores above or below the knockout--control threshold were classified as transcriptionally
predisposed toward different downstream fates.

To evaluate the quality of these predictions, we trained a lasso logistic regression classifier to
assess whether the predicted fate could be distinguished from the true fate labels (GC and PB
cells from the single-cell multi-ome dataset). Two scenarios were defined: first, if a predicted
fate could not be distinguished from the true identity based on transcriptional features, the cell
was assigned to the same identity. Conversely, if the predicted fate could be confidently
discriminated from the true identity, we inferred that FocalFire did not specify the cell as
belonging to the predicted fate. This framework produced classification-loss scores that
quantified the predictive confidence of rollback analysis, thereby enabling evaluation of the
model's performance in capturing early fate bias.


Gene set enrichment analysis using ESCAPE
-----------------------------------------

To interrogate transcriptional heterogeneity and quantify the enrichment of gene-expression
signatures at the single-cell level, we employed the ESCAPE (Enrichment of Single Cell Analysis by
Pathway Expression) framework (Borcherding et al., Bioconductor, `bioc.org/packages/escape <https://bioconductor.org/packages/escape>`_).
ESCAPE is a single-cell gene set enrichment analysis tool specifically designed to leverage the
high dimensionality and variability inherent to scRNA-seq data. By integrating predefined gene
sets or custom marker lists, ESCAPE calculates enrichment scores for individual cells, enabling
sensitive detection of transcriptional programs and cellular states within heterogeneous
populations. In our study, we applied ESCAPE to assess whether the transcriptional profiles of
predicted "KO-like" subsets reflected expected lineage biases. Enrichment was calculated using
canonical marker genes representative of progenitor-like exhausted T cells (Tpex: *Tcf7*,
*Slamf6*, *Myb*, *Sell*, *Bach2*) and terminally exhausted T cells (Tex: *Havcr2*, *Pdcd1*,
*Entpd1*, *Cd38*, *Cd244a*, *Cxcr6*, *Ifng*). These analyses allowed us to systematically compare
enrichment patterns across subsets and gain insights into the differentiation status and
functional states of the populations under investigation.


Assessing chromatin co-regulation for TF pairs using Cicero co-accessibility analysis
--------------------------------------------------------------------------------------

To investigate whether transcriptional regulation within cellular regulons is supported by
chromatin accessibility, we performed a co-accessibility and TF motif enrichment analysis using
Cicero, applied to regulon genes. For each gene within a regulon, we first identified accessible
peaks co-accessible with the gene's promoter region using Cicero, focusing on peaks with high
co-accessibility scores. We then scanned these co-accessible peaks for the presence of binding
motifs associated with TF pairs that have high enrichment scores based on the fusion GRN. The
following steps were carried out. First, for each regulon gene, enriched TF pairs identified as
co-regulators from the fusion GRN were selected. These TF-pair motifs were scanned within the set of
co-accessible peaks for the gene's promoter. Peaks containing TF motifs were retained, and the
mean co-accessibility score of these motif-containing peaks was recorded for each gene. Second,
as a control, size-matched TF pairs were generated based solely on GRN topology, that is, TFs
connected within the inferred regulatory network independent of regulon context. For each gene,
these TF pairs were analyzed using the same workflow---motif scanning over co-accessible
promoter peaks followed by co-accessibility score calculation. Third, a second size-matched
control was created using random TF pairs. Random pairings of TFs, not derived from the network or
regulons, were similarly scanned for motifs in the co-accessible promoter regions of each gene,
and the average co-accessibility of motif-containing peaks was computed.

The distribution of average co-accessibility scores for each group---regulon-enriched TF pairs,
topological control pairs, and random controls---was visualized using boxplots. This analysis
allowed us to assess whether co-regulatory TF pairs in our GRNs showed higher chromatin
co-accessibility in a motif-specific manner, providing orthogonal support for inferred
transcriptional control.
