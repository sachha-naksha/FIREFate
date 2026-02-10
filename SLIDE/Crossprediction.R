########################################
##        Cross prediction     ##
########################################
library(pROC)
library(SLIDE)
################
# Train Data
################
## orig_x and orig_y are X and Y matrices that SLIDE model has been trained on
orig_path <- ".."
orig_x <- read.csv(paste0(orig_path,"X_train.csv"), row.names = 1, header = TRUE)
orig_y <- read.csv(paste0(orig_path,"Y_train.csv"), row.names = 1, header = TRUE)
################
# Test Data #
################
## new_x and new_y are X and Y matrices that we want to test the SLIDE model on 
new_path <- ".."
new_x <- read.csv(paste0(new_path,"/X_test.csv"), row.names = 1, header = TRUE)
new_y <- read.csv(paste0(new_path,"/Y_test.csv"), row.names = 1, header = TRUE)
################
# SLIDE Model #
################
# These necessary files are available in the best SLDIE model out folder
path_SLIDE <- ".."
LFs <- readRDS(paste0(path_SLIDE,"/AllLatentFactors.rds"))
orig_z <- read.csv(paste0(path_SLIDE,"/z_matrix.csv", row.names = 1, header = TRUE))
slide_res <- readRDS(paste0(path_SLIDE,"/SLIDE_LFs.rds"))
########################################
A_mat = LFs$A
beta = LFs$beta
sig_lfs = as.numeric(stringr::str_replace(c(slide_res$marginal_vals,slide_res$interaction$p2),
                                          pattern = "z", replacement = ""))

sig_lfs = paste0("Z", c(slide_res$marginal_vals,slide_res$interaction$p2))
sig_lfs = unique(sig_lfs)
non_sig_lf_cols = which(!colnames(A_mat) %in% sig_lfs)
beta[non_sig_lf_cols] = 0


# get the genes from Train X matrix that are also in Test X matrix
orig_in_new = colnames(orig_x)[which(colnames(orig_x) %in% colnames(new_x))]
length(orig_in_new) 

orig_not_in_new = colnames(orig_x)[which(!colnames(orig_x) %in% colnames(new_x))]
length(orig_not_in_new) 

orig_x[, orig_not_in_new] <- 0

new_not_in_orig = colnames(new_x)[which(!colnames(new_x) %in% colnames(orig_x))]
length(new_not_in_orig) 

new_x = new_x[, !colnames(new_x) %in% new_not_in_orig]

if (length(orig_not_in_new) > 0) {
  new_x <- cbind(new_x, matrix(0, nrow(new_x), length(orig_not_in_new)))
  colnames(new_x)[(ncol(new_x) - length(orig_not_in_new) + 1):ncol(new_x)] <- orig_not_in_new
}

new_x_worg = as.matrix(new_x[, match(colnames(new_x), colnames(orig_x))])
orig_x_worg = as.matrix(orig_x[, match(colnames(orig_x), colnames(orig_x))])

# get new predicted Z
orig_z = SLIDE::predZ(orig_x_worg , LFs)
colnames(orig_z) <- paste0("Z", c(1:ncol(orig_z)))

new_z = SLIDE::predZ(new_x_worg, LFs)
colnames(new_z) <- paste0("Z", c(1:ncol(new_z)))

orig_input <- orig_z[ , sig_lfs ]
new_input <-  new_z[ , sig_lfs]
###########################################################################
# Fit a linear model to find the predicted y

lin_reg <- stats::glm(orig_y$V1 ~ ., data = as.data.frame(orig_input), family = "gaussian")

orig_y_pred   <- predict(lin_reg, as.data.frame(orig_input) ,type = 'response')
orig_auc <- pROC::auc(response=as.matrix(orig_y), predictor=as.matrix(orig_y_pred),quite=T)
print(orig_auc)

new_y_pred   <- predict(lin_reg, as.data.frame(new_input) ,type = 'response')
new_auc <- pROC::auc(response=as.matrix(new_y), predictor=as.matrix(new_y_pred),quite=T)
print(new_auc) 

###########################################################################
# Plot the ROC curve for the original and new data
roc_orig <- pROC::roc(response=as.matrix(orig_y), predictor=as.matrix(orig_y_pred))
roc_new <- pROC::roc(response=as.matrix(new_y), predictor=as.matrix(new_y_pred))

# Plot both ROC curves
plot(roc_orig, col="#7BDE7B", lwd=2)
lines(roc_new, col="#B83636", lwd=2)
legend("bottomright", legend=c(paste("Original AUC =", round(orig_auc, 4)), 
                               paste("New AUC =", round(new_auc, 4))), col=c("#7BDE7B", "#B83636"), lwd=2)


