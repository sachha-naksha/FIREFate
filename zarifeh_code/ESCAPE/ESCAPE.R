################################################################   
## ESCAPE: Easy single cell analysis platform for enrichment
################################################################ 
## Install the libraries if requiered
if (!require("BiocManager", quietly = TRUE))
install.packages("BiocManager")
BiocManager::install("escape")

if (!require(effsize)) install.packages("effsize")   # For Cliff's delta

library(effsize)
library(scRepertoire)
library(GSVA)
library(escape)
library(Seurat)
library(dplyr)
library(tidyr)
library(ggplot2)

# Path to the seurat object which contains single cell RNA-seq data of cells of interest
path <- "..."
scRep_data <- readRDS(paste0(path, "file.rds"))

# We also need a set of marker genes to do the enrichment based on those 
gene.sets <- list(State1_markers = c("marker1_1", "marker1_2", ...),
                  State2_markers = c("marker2_1", "marker2_2", ...)
                 )

# Apply a function to each gene set to keep only genes that are present in scRep_data
matched <- lapply(gene.sets, function(gs) intersect(gs, rownames(scRep_data)))
matched

#Running Enrichment and also automatically amend the single-cell object with the values added as an assay, which 
#is named via the new.assay.name parameter. This facilitates easy downstream visualization and analysis.

scRep_data <- runEscape(scRep_data, 
                           method = "ssGSEA",
                           gene.sets = gene.sets, 
                           groups = 100, 
                           min.size = 3,
                           new.assay.name = "escape.ssGSEA")

# escape_scores matrix contains scores for each single cell corresponds to the state1 and state2
escape_scores <- GetAssayData(scRep_data, assay = "escape.ssGSEA", slot = "data")

################################################################
### Visualization: dot plots
################################################################
# Create the plots with consistent ylim and larger text
p1 <- geyserEnrichment(scRep_data, 
                       assay = "escape.ssGSEA",
                       gene.set = "State1-markers",
                       group.by  = "orig.ident",
                       palette   = "Blue-Yellow") + 


p2 <- geyserEnrichment(scRep_data, 
                       assay = "escape.ssGSEA",
                       gene.set = "State2-markers",
                       group.by  = "orig.ident",
                       palette   = "Red-Blue") +

# Combine plots
p1 + p2
################################################################
# Calculating Cliff's delta values
################################################################
# X1 and X2 is defined to separate the scores of cells for each state
X1 <- as.numeric(escape_scores[1, ])
X2 <- as.numeric(escape_scores[2, ])

cd <- cliff.delta(X1, X2)

# calculating p-value using Wilcoxon rank-sum test ---
wilcox_res <- wilcox.test(X1, X2, exact = FALSE)
pval <- wilcox_res$p.value
pval
