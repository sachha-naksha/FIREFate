Methods
=======

**FIREFate** is a framework that focuses dense mechanistic models of gene regulation, through
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

We implemented and tested FIREFate in Python (versions xx) and designed it for use in a Jupyter
Notebook environment. FIREFate code is open source and available on GitHub at
`github.com/sachha-naksha/FIREFate <https://github.com/sachha-naksha/FIREFate>`_ and
`github.com/xxx <https://github.com/xxx>`_, along with detailed
function descriptions and tutorials at ``<READ-THE-DOCS>``. Additionally, we provide a
user-friendly web application at
`pitt-csi.shinyapps.io/firefate <https://pitt-csi.shinyapps.io/firefate/>`_ that facilitates
FIREFate analyses and enables
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
(`github.com/jishnu-lab/SLIDE <https://github.com/jishnu-lab/SLIDE>`_) and the FIREFate GitHub repository
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


Benchmarking FIREFate cellular programs
---------------------------------------

To enable direct comparison of order-1 transcriptional programs enriched by FIREFate, namely
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

Quantification of local TF activity across episodes of regulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The trajectory branches for both datasets were broken into episodes of invariant TF activity. T
cells have a linear trajectory divided into [N] episodes, while B cells follow a bifurcating
trajectory with [N] episodes per branch. Each episode was defined as a contiguous slice of
equidistantly sampled pseudotime windows from the trajectory. Episodes are non-overlapping, and
their boundaries were placed at approximately equidistant intervals across the total number of
sampled pseudotime points (:math:`n=40`), yielding episode-specific subnetworks that capture
temporally coherent regulatory states.

For each episode, a weighted regulatory episodic network was constructed by chunking the
smoothed network statistics (``stat.net``, ``stat.fbinarize``) at equidistantly sampled pseudotime
points within the episodic duration (:math:`n=5`). Regulatory edge weights (:math:`\beta`
coefficients) were retrieved for all TF--target pairs active within the episode. Edges were
subsequently filtered using two criteria applied in parallel: (i) a minimum number of non-zero
timepoint threshold (:math:`\geq 3`), ensuring the regulatory interaction was consistently
observed across the episode; and (ii) a one-sample :math:`t`-test against zero on the non-zero
:math:`\beta` values (:math:`\alpha = 0.05`), retaining only edges with statistically significant
regulatory signal. Additionally, direction invariance was enforced, requiring that all non-zero
:math:`\beta` values for a given TF--target pair share the same sign within the episode, thus
restricting retained edges to those exerting a consistent activating or repressing influence
throughout the episode.

For each sampled pseudotime window within an episode, a regulatory force metric was calculated to
weight each TF--target edge by the transcriptional activity of the cognate TF. Specifically, the
force for edge :math:`(i,j)` at pseudotime point :math:`t` was defined as:

.. math::
   :label: eq-force-metric

   F(i,j,t) = \mathrm{sign}(\beta_{i,j,t}) \cdot \exp\left(\log_{10}\left(|\beta_{i,j,t}| + \epsilon\right) + \log_{10}\left(X_{i,t} + \epsilon\right)\right)

where :math:`\beta_{i,j,t}` is the regulatory coefficient for TF :math:`i` on target :math:`j` at
time :math:`t`, :math:`X_{i,t}` is the smoothed log-CPM expression of TF :math:`i` at time
:math:`t`, and :math:`\epsilon = 10^{-10}` is a small constant used to avoid numerical instability.
The average force across all timepoints within the episode was computed per edge, and the top 2% of
edges ranked by absolute average force were retained to form the episodic GRN.

To identify TFs whose targets were enriched for genes of interest, here late-fate or lineage-fate
(LF) genes, a hypergeometric enrichment test was performed for each TF within each episodic GRN.
For a given TF with :math:`n` downstream targets in the episodic GRN, the probability of observing
:math:`k` or more LF-gene targets by chance was calculated as:

.. math::
   :label: eq-hg-episode

   P(X \geq k) = 1 - \mathrm{CDF}_{\mathrm{hypergeom}}(k - 1; N, K, n)

where :math:`N` is the total number of genes in the episodic GRN, :math:`K` is the total number of
LF genes active in the episode, and :math:`n` is the total number of targets regulated by the TF.
An enrichment score was computed as the fold enrichment over expectation,
:math:`k / (n \cdot K / N)`. TFs were ranked by enrichment score within each episode, and those
with zero enrichment were excluded. This framework enabled systematic, episode-resolved
identification of TFs whose downstream targets are disproportionately composed of
fate-determining genes, providing a temporally stratified view of regulatory control across the
differentiation trajectory.

Analysis of global TF activity across trajectories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analysis of global TF activity across trajectories was performed to summarize higher-level
regulatory trends across the inferred developmental paths.


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
system. FIREFate projected these programs onto ABCs (days 2 and 4) from single-cell multi-ome
data to estimate their transcriptional predisposition toward one of two downstream fates (GC or
PB). The latent cellular programs defined a score for each cell, representing the degree to which
its expression profile aligned with either the knockout or control state. Using these scores, we
assessed the discriminative power of each model by examining how well the score distributions
separated knockout and control cell populations. For each model, we defined an optimal threshold
based on the distribution that most effectively distinguished these two groups. We then applied
the same trained models to the day 2 and day 4 ABC multi-ome data using the ``predZ`` function in
the SLIDE package, generating predicted scores for uncommitted cells. Using the thresholds defined
from the Perturb-seq models, FIREFate assigned a predicted fate score to each ABC. Cells with
predicted scores above or below the knockout--control threshold were classified as transcriptionally
predisposed toward different downstream fates.

To evaluate the quality of these predictions, we trained a lasso logistic regression classifier to
assess whether the predicted fate could be distinguished from the true fate labels (GC and PB
cells from the single-cell multi-ome dataset). Two scenarios were defined: first, if a predicted
fate could not be distinguished from the true identity based on transcriptional features, the cell
was assigned to the same identity. Conversely, if the predicted fate could be confidently
discriminated from the true identity, we inferred that FIREFate did not specify the cell as
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
