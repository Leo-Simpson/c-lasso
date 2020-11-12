# Go from phyloseq object to something ready to input into trac

library(phyloseq)
library(Matrix)
library(tidyverse)
library(trac)

tara <- readRDS("tara_10.RDS")

# Use salinity as outcome
y <- sample_data(tara)$Mean_Salinity..PSU.
summary(y)
keep <- which(!is.na(sample_data(tara)$Mean_Salinity..PSU.))

# Longhurst provinces in Longhurst
sample_data(tara)$Marine.pelagic.biomes..Longhurst.2007.

tax <- tara@tax_table@.Data

# replace "unclassified" with the appropriate blank tag
blank <- paste0(c("k", "p", "c", "o", "f", "g", "s"), "__")
for (i in 1:7) tax[tax[, i] == "unclassified", i] <- blank[i]
for (i in 1:7) tax[tax[, i] == "undef", i] <- blank[i]
for (i in 1:7) tax[tax[, i] == "", i] <- blank[i]

tax <- cbind("Life", tax); colnames(tax)[1] <- "Rank0"
# add an OTU column
tax <- cbind(tax, rownames(tax))
colnames(tax)[ncol(tax)] <- "OTU"

colnames(tax)[(colnames(tax) == "OTU.rep")] <- "Species"
colnames(tax)[(colnames(tax) == "Domain")] <- "Kingdom"

# make it so labels are unique
for (i in seq(2, 8)) {
  # add a number when the type is unknown... e.g. "g__"
  ii <- nchar(tax[, i]) == 3
  if (sum(ii) > 0)
    tax[ii, i] <- paste0(tax[ii, i], 1:sum(ii))
}

# cumulative labels are harder to read but easier to work with:
for (i in 2:9) {
  tax[, i] <- paste(tax[, i-1], tax[, i], sep = "::")
}
tax <- as.data.frame(tax)

# form phylo object:
tree1 <- tax_table_to_phylo(~Rank0/Kingdom/Phylum/Class/Order/Family/Genus/Species/OTU,
                            data = tax, collapse = TRUE)

# convert this to an A matrix to be used for aggregation:
A <- phylo_to_A(tree1)

dat <- list(y = y[keep],
            x = t(tara@otu_table@.Data)[keep,],
            tree = tree1,
            tax = tax,
            A = A,
            sample_data = as_tibble(sample_data(tara)[keep,]))
# rows of A correspond to OTUs as do columns of x
# rearrange columns of x to be in the order of rows of A:
dat$x <- dat$x[, match(str_match(rownames(A), "::([^:]+)$")[, 2],
                       colnames(dat$x))]
identical(str_match(rownames(A), "::([^:]+)$")[,2],
          colnames(dat$x))
saveRDS(dat, file = "tara_sal_processed.RDS")


