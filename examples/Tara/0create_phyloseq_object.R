# Project: Tree-aggregation of compositional data
# Import and prune Tara 

library(phyloseq)

load("original/taraData.rda")

tara  <- physeq

map <- data.frame(metadata, row.names=metadata$SampleName)

sample_data(tara) <- map ## assign metadata to phyloseq object

tara
# phyloseq-class experiment-level object
# otu_table()   OTU Table:         [ 35651 taxa and 139 samples ]
# sample_data() Sample Data:       [ 139 samples by 53 sample variables ]
# tax_table()   Taxonomy Table:    [ 35651 taxa by 7 taxonomic ranks ]

## Save data to RDS file (all ocean taxa present)
saveRDS(tara, file = "tara.RDS")

## Prune samples

badTaxa = c("OTU1") # undefined across all ranks
allTaxa = taxa_names(tara)
allTaxa <- allTaxa[!(allTaxa %in% badTaxa)]
tara = prune_taxa(allTaxa, tara)

depths <- colSums(tara@otu_table@.Data) ## calculate sequencing depths

## Pruning (Minimum sequencing depth: at least 10000 reads per sample)
tara.filt1 <- prune_samples(depths > 10000, tara) 
tara.filt1
# phyloseq-class experiment-level object
# otu_table()   OTU Table:         [ 35650 taxa and 139 samples ]
# sample_data() Sample Data:       [ 139 samples by 53 sample variables ]
# tax_table()   Taxonomy Table:    [ 35650 taxa by 7 taxonomic ranks ]

## Pruning (taxa present in at least 1% of samples)
freq <- rowSums(sign(tara.filt1@otu_table@.Data))

## Pruning (taxa present in at least 10% of samples)
tara.filt4 <- prune_taxa(freq > 0.1 * nsamples(tara.filt1), tara.filt1) 
tara.filt4
# phyloseq-class experiment-level object
# otu_table()   OTU Table:         [ 8916 taxa and 139 samples ]
# sample_data() Sample Data:       [ 139 samples by 53 sample variables ]
# tax_table()   Taxonomy Table:    [ 8916 taxa by 7 taxonomic ranks ]


## Save data to RDS file (taxa present in at least 10% of samples)
saveRDS(tara.filt4,file = "tara_10.RDS")

