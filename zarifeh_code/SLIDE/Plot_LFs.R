####################################################################################
##  Scatter plots to show the power of SLIDE LFs in State discrimination ##
####################################################################################

library(ggplot2)
library(dplyr)

# Path to X and Y (The ones you have run SLIDE on)
x_path <- ".../Data/Concat/Male_Donor/X_GC_PB.csv"
y_path <- ".../Data/Concat/Male_Donor/Y_GC_PB.csv"

# Update this with the correct path: this path contains the best SLIDE results (Z matrix and SLIDE_LFs.rds)
out_path <- ".../out/0.3_1_out/"  

X <- read.csv(x_path, row.names = 1, header = TRUE)
Y <- read.csv(y_path, row.names = 1, header = TRUE)


Z <- read.csv(paste0(out_path, "z_matrix.csv"), row.names = 1, header = TRUE)
slide_res <- readRDS(paste0(out_path, "SLIDE_LFs.rds"))

marginal_LFs <- slide_res$SLIDE_res$marginal_vars
marginal_LFs
interaction_LFs <- slide_res$interaction$p2
interaction_LFs

# Loop over each pair of marginal LFs for (marginal_LFs, marginal_LFs)
for (i in marginal_LFs) {
  for (j in marginal_LFs) {
    if (i != j) {  # Skip the case where i == j
      Z_x <- Z[, i]
      Z_y <- Z[, j]
      
      p <- ggplot(X, aes(x = Z_x, y = Z_y, color = Y == 1)) +
        geom_point() +
        xlab(paste0("Z", i)) +
        ylab(paste0("Z", j)) +
        scale_color_manual(values = c("#7BDE7B", "#B83636")) +
        theme(
          panel.background = element_blank(),
          panel.border = element_blank(),
          axis.line = element_line(color = "black"),
          axis.ticks = element_line(color = "black"),
          axis.text = element_text(color = "black")
        )
      
      # Save the plot as SVG
      ggsave(
        filename = file.path(out_path, paste0("marginal_LF_", i, "_vs_", j, ".svg")),
        plot = p,
        device = "svg",
        width = 8,  # Set the width of the plot
        height = 6  # Set the height of the plot
      )
    }
  }
}


# Loop over each pair of (interaction_LFs, marginal_LFs) for (x, y)
for (i in marginal_LFs) {
  for (j in interaction_LFs) {
    if (i != j) {  # Skip the case where i == j
      Z_x <- Z[, i]
      Z_y <- Z[, j]
      
      p <- ggplot(X, aes(x = Z_x, y = Z_y, color = Y == 1)) +
        geom_point() +
        xlab(paste0("Z", i)) +
        ylab(paste0("Z", j)) +
        scale_color_manual(values = c("#7BDE7B", "#B83636")) +
        theme(
          panel.background = element_blank(),
          panel.border = element_blank(),
          axis.line = element_line(color = "black"),
          axis.ticks = element_line(color = "black"),
          axis.text = element_text(color = "black")
        )
      
      # Save the plot as SVG
      ggsave(
        filename = file.path(out_path, paste0("SigZ", i, "_Interaction_Z", j, ".svg")),
        plot = p,
        device = "svg",
        width = 8,  # Set the width of the plot
        height = 6  # Set the height of the plot
      )
    }
  }
}

