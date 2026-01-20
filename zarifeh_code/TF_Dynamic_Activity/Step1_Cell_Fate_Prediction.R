######################################################################################################################
##  This code contains steps we need to do preprocessing of the perturb data, running SLIDE ##
## then superimposition of LFs from SLIDE models on TF perturb-seq data on unperturbed cells ## 
######################################################################################################################

#########################################################################################################################################
## Step1: 
## Each TF-Perturb Data is first preprocessed and concatenated with NTC (non targeted cells) the same way as for unperturbed data
## you can use Steps in preprocessing folder, So you will have "X_TFname_NTC.csv" for each TF
#########################################################################################################################################

#########################################################################################################################################
## Step2:
## Second Step is using SLIDE models to predict the cell lineage of ABCs from unperturbed data
### so for running SLIDE and using that we will find the shared genes of each TF-Perturb vs. NTC with ABCs and then run SLIDE on
#########################################################################################################################################
# Define output path to save the figures and resulots
out_path <- "..."
# path to unperturbed dataset
input_path1 <- "..."
# path to perturb dataset
input_path2 <- "..."

#unperturbed_x is activated B cells from unperturbed datasets
unperturbed_x <- read.csv(paste0(input_path1,"/unperturbed_x.csv"), row.names = 1)

# TF_x is your preprocessed and concatenated Tf-perturb and NTCs
TF_x <- read.csv(paste0(input_path2,"/X_TFname_NTC.csv"), row.names = 1)

shared_columns <- intersect(colnames(TF_x), colnames(unperturbed_x))

# Create subsets containing only the shared columns
train_x <- TF_x[, shared_columns]
val_x <- unperturbed_x[, shared_columns]

# writing for each TF 
write.csv(train_x,paste0(out_path,"/X_train.csv"), row.names = TRUE)
write.csv(val_x,paste0(out_path,"/X_val.csv"), row.names = TRUE)

# train_y contains 1 for cells from TF perturb-seq and 0 for NTC
# This y can be generated using Steps in preprocessing folder
train_y <- read.csv(paste0(input_path2,"/Y_tarin.csv"), row.names = 1, header = TRUE)
################################################################################
#########################################################################################################################################
## Step3: 
## Third Step is using SLIDE models and get the best model that discriminate TF-perturb vs NTCs
#########################################################################################################################################
# These files are the outputs of your best SLIDE model 
SLIDE_path <- "..."
AllLFs <- readRDS(paste0(SLIDE_path,"/AllLatentFactors.rds"))
slide_res <- readRDS(paste0(SLIDE_path,"/SLIDE_LFs.rds"))
z_matrix <- read.csv(paste0(SLIDE_path,"/z_matrix.csv"),row.names = 1, header = TRUE )
################################################################################
#########################################################################################################################################
## Step4: Once you get the best SLIDE model, you will find one significant LF that can discriminate TF-perturb cells and NTCs very well
## We use that sigZ and superimpose it to unperturbed cells to see if we can predict their fate
#########################################################################################################################################
library(SLIDE)
library(ggplot2)
library(dplyr)
library(pROC)
################ predZ function##############
predZ <- function(x, er_res) {
  A_hat <- er_res$A
  C_hat <- er_res$C
  Gamma_hat <- er_res$Gamma
  Gamma_hat <- ifelse(Gamma_hat == 0, 1e-10, Gamma_hat)
  Gamma_hat_inv <- diag(Gamma_hat ** (-1))
  G_hat <- crossprod(A_hat, Gamma_hat_inv) %*% A_hat + solve(C_hat)
  Z_hat <- x %*% Gamma_hat_inv %*% A_hat %*% MASS::ginv(G_hat)
  return (Z_hat)
}
#############################################
train_in_val <- colnames(train_x)[which(colnames(train_x) %in% colnames(val_x))]
train_in_val = colnames(train_x)[which(colnames(train_x) %in% colnames(val_x))]
train_not_in_val = colnames(train_x)[which(!colnames(train_x) %in% colnames(val_x))]
val_x_worg = as.matrix(val_x[, match(colnames(val_x), colnames(train_x))])
train_x_worg = as.matrix(train_x[, match(colnames(train_x), colnames(train_x))])

val_x <- as.matrix(val_x_worg)
train_x <- as.matrix(train_x_worg)

val_x <-scale(val_x)
val_z <- predZ(val_x, AllLFs)
colnames(val_z) <- paste0("Z", c(1:ncol(val_z)))


sigZ = Z5  #here you should specify the significant standalone LFs
sigZ_index = 5

x_limits = c(-3, 3)
y_limits = c(0, 1000)
p1 <- ggplot(z_matrix, aes(x = sigZ )) + 
  geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(
    theme_minimal(base_size = 15)) +  # Clean and minimal theme with larger text
    coord_cartesian(xlim = x_limits, ylim = y_limits) +
      theme(plot.title = element_text(hjust = 0.5),
            panel.background = element_blank(),
            panel.grid = element_blank())  # Remove the grid

# Save as PDF
ggsave(filename = file.path(out_path, "SigZ_distribution.pdf"), plot = p1, width = 8, height = 6, units = "in")

p2 <-ggplot(val_z, aes(x = sigZ)) +# here you should specify the significant standalone LFs
  geom_histogram(binwidth = 0.1, fill = "skyblue", color = "black", alpha = 0.7) +
  labs(
    theme_minimal(base_size = 15)) +  # Clean and minimal theme with larger text
    coord_cartesian(xlim = x_limits, ylim = y_limits) +
      theme(plot.title = element_text(hjust = 0.5),
            panel.background = element_blank(),
            panel.grid = element_blank())  # Remove the grid

# Save as PDF
ggsave(filename = file.path(out_path, "predZ_distribution.pdf"), plot = p2, width = 8, height = 6, units = "in")

threshold = -0.3 # specify the value based on SLIDE results

per <- val_z[val_z[,sigZ_index] < threshold,, drop= FALSE][,1]
length(per)
print(length(per)/length(val_z[,sigZ_index])*100) #Percentage of cells from unperturbed cells in val_x that shows late state lineage commitment

# you can save those cells
write.csv(per, paste0(out_path,"/early_perturb.csv"), row.names = TRUE)

# You can also have the gene expression matrix of those cells
val_x <- read.csv(paste0(out_path,"/X_val.csv"), row.names = 1)
per_cells_rownames <- rownames(per)
Per_GEX <- val_x[rownames(val_x) %in% per_cells_rownames, ]
write.csv(Per_GEX,paste0(out_path, "per_GEX.csv"), row.names = TRUE)

###################################################################################################################################################
## Step5:
##Having identified perturbation-like cells as a subset of the unperturbed population, we next assess whether they resemble either bifurcated fate.
##To do so, we compare these cells to bona fide terminal-state cells from the unperturbed dataset. We first identify genes shared between
##perturbation-like cells and bifurcated states, concatenate the corresponding expression profiles, and then fit a logistic regression model 
##to quantify similarity to each bifurcated state.
###################################################################################################################################################
# Load the datasets: X1 and X2 are the bifurcated cells in state 1 and state 2
X1 = read.csv(paste0(input_path1, "/X1.csv"))
X2 = read.csv(paste0(input_path1, "/X2.csv"))
#for each TF perturb data, we will find the shared genes with X1 and X2
datasets <- list(X1 = X1, X2 = X2)
# Iterate through each dataset
for (name in names(datasets)) {
  X <- datasets[[name]]
  # Find shared columns between X and val_x
  shared_columns <- intersect(colnames(X), colnames(val_x))
  # Create subsets containing only the shared columns
  X_subset <- X[, shared_columns]
  X_subset <- scale(X_subset)
  # Write the scaled subset to a CSV file
  write.csv(X_subset, paste0(out_path, "/TF_", name, ".csv"), row.names = TRUE)
}

## Now we concatenate the X1 and X2 with Per_GEX we got from step 4 for each of the TFs

# Iterate through each dataset
for (name in names(datasets)) {
  X1 <- Per_GEX
  X2 <- datasets[[name]]
  
  # Find shared columns
  common_cols <- intersect(colnames(X1), colnames(X2))
  
  # Concatenate the two datasets based on shared columns
  X <- rbind(X1[, common_cols], X2[, common_cols])
  write.csv(X, paste0(out_path, "/X_TF_", name, ".csv"), row.names = TRUE)
  
  # Create a label matrix Y
  Y <- matrix(0, nrow = nrow(X), ncol = 1)
  rownames(Y) <- rownames(X)
  
  # Assign labels: 1 for X1 rows, 0 for X2 rows
  Y[rownames(X1), ] <- 1
  Y[rownames(X2), ] <- 0
  write.csv(Y, paste0(out_path, "/Y_TF_", name, ".csv"), row.names = TRUE)
}

################################
