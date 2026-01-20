##################################################################
     ##  CV Boxplot to show the SLIDE model performance ##
##################################################################

library(ggplot2)

# Define the path to the folders containing the .rds files
base_path <- ".../slidecv"  # Replace with the path to your directories
folders <- list.dirs(base_path, recursive = FALSE)

# Initialize an empty data frame to store the AUC values
auc_data <- data.frame(Method = character(), AUC = numeric(), stringsAsFactors = FALSE)
# Set a path to save the boxplot
pdf(".../slidecv/SLIDECV_Boxplot.pdf", height = 5, width = 5)

# Loop through each replicate folder
for (folder in folders) {
  # Get a list of .rds files in the current folder
  rds_files <- list.files(folder, pattern = "\\.rds$", full.names = TRUE)
  
  # Loop through each .rds file in the current folder
  for (file in rds_files) {
    # Load the .rds file
    result <- readRDS(file)
    
    # Extract AUC values for SLIDE and SLIDE_y
    slide_auc <- result[["final_corr"]][result[["final_corr"]]$method == "SLIDE", "auc"]
    slide_y_auc <- result[["final_corr"]][result[["final_corr"]]$method == "SLIDE_y", "auc"]
    
    # Append to the data frame
    auc_data <- rbind(auc_data, data.frame(Method = "Actual", AUC = slide_auc))
    auc_data <- rbind(auc_data, data.frame(Method = "Permutation", AUC = slide_y_auc))
  }
}

ggplot(auc_data, aes(x = Method, y = AUC, fill = Method)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "",
       x = "Method",
       y = "AUC") +
  scale_fill_manual(values = c("Actual" = "#007FFF", "Permutation" = "skyblue")) +
  theme(panel.grid = element_blank(), 
        axis.line.y = element_line(color = "black"), # Add border line on y-axis
        axis.ticks.y = element_line(color = "black"),
        axis.line.x = element_line(color = "black"), # Add border line on y-axis
        axis.ticks.x = element_line(color = "black"))
dev.off()
