##############################################################################################################
# Concat Two GEX matrices to make the X matrix ready for SLIDE analysis
# Preprocessing gene expressiong matrix also can be done via the preprocessing steps available in 
# SLIDE method github: https://github.com/jishnu-lab/SLIDE/blob/main/vignettes/Preprocessing-and-Filtering.Rmd
##############################################################################################################

# Define input and output folders and file names
input_folder <- "..."
output_folder <- "..."
dir.create(output_folder, recursive = TRUE) 

# Define the file names that you need to concat
file_X1 <- "X1.csv"
file_X2 <- "X2.csv"

# Define the file names for X and Y after concatenation
output_X <- "X.csv"
output_Y <- "Y.csv"


# Read the csv files as data frames
X1 <- read.csv(paste0(input_folder, file_X1), row.names = 1, header = TRUE) 
X2 <- read.csv(paste0(input_folder, file_X2), row.names = 1, header = TRUE) 

# Check dimensions
dim(X1) 
dim(X2)

# Find the common column names between X1 and X2
common_cols <- intersect(colnames(X1), colnames(X2))  
length(common_cols) 

# Remove common columns from X1 and X2
X1_new <- X1[, setdiff(colnames(X1), common_cols)]
X2_new <- X2[, setdiff(colnames(X2), common_cols)]

# Merging not shared columns
X <- merge(X1_new, X2_new, all=TRUE, by=0)

# Set the Row.names column as row names
rownames(X) <- X$Row.names
X$Row.names <- NULL

# Add columns for common genes initialized to 0
for (col in 1:length(common_cols)) {
  X <- cbind(X, 0)
}

# Rename the last columns for common genes
colnames(X)[(ncol(X)-length(common_cols)+1):ncol(X)] <- paste0(common_cols)

# Inserting shared genes
v <- rbind(X1[, common_cols], X2[, common_cols])

# Get the row indices of X matching the row names of v
indices <- match(rownames(X), rownames(v))

# Reorder the rows of v
v_sorted <- v[indices, ]
X[, common_cols] <- v_sorted

# Replace NA values with 0
X[is.na(X)] <- 0

# Check final dimensions
dim(X) 

# Writing CSV
write.csv(X, file = paste0(output_folder, output_X), row.names = TRUE, col.names = TRUE)

#####################################################
# Making Y (0 for file_X1 and 1 for file_X2)
# Create an empty matrix with the same number of rows as X
Y <- matrix(0, nrow = nrow(X), ncol = 1)
rownames(Y) <- rownames(X)

# Assign 1s to the rows corresponding to X2
Y[rownames(X2), ] <- 1
Y[rownames(X1), ] <- 0

# Writing CSV with Y matrix
write.csv(Y, file = paste0(output_folder, output_Y), row.names = TRUE, col.names = TRUE)

