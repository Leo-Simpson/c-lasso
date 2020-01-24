Description of subset of COMBO data 

The present data is a subset of the COMBO study which was first analyzed in: 

Wu, G. D., Chen, J., Hoffmann, C., Bittinger, K., Chen, Y.-Y., Keilbaugh, S. A., … Lewis, J. D. (2011). 
Linking Long-Term Dietary Patterns with Gut Microbial Enterotypes. 
Science, 334(6052), 105–108. http://doi.org/10.1126/science.1208344

The primary data is not readily available and the supplementary information of the paper does not give enough 
detail to replicate the study. 

The data in this folder was provided by Pixu Shi in November 2017. It comprises processed data, including data used in:

Pixu Shi, Anru Zhang, and Hongzhe Li
Regression analysis for microbiome compositional data
Ann. Appl. Stat., Volume 10, Number 2 (2016), 1019-1040.

The folder contains the following files and description:  

Unfiltered data

n=96 samples (patients)
p=87 bacterial genera (processed, no primary OTU data available)  

BMI.csv		Outcome:  	nx1 vector of Body Mass Index 	   CaloriData.csv	Covariate 1: 	nx1 vector of calorie intakeFatData.csv	Covariate 2:	nx1 vector of fat intake

GeneraPhylo.csv		p x 7 matrix of bacterial phylogenic identity
			OTU_i ’Domain','Phylum','Class','Order','Family','Genus' 
GeneraCounts.csv	p x n matrix of Genera (OTU counts) Filtered data used in Shi et al. study

n=96 samples (patients)
pf=45 bacterial genera (processed, subset of previous data)  
GeneraFilteredPhylo.csv	pf x 7 matrix of bacterial phylogenic identity
			 	OTU_j ’Domain','Phylum','Class','Order','Family','Genus' 
GeneraFilteredCounts.csv	pf x n matrix of Genera (OTU counts)

MappedIndices.csv		Indices containing the mapping of genera from full data to filtered data


C_Filtered.csv			pf+1 x 5 binary matrix: First column Genera Name, First row: Phylum 
				Remaining matrix comprises matrix C used in Shi et al in Sec. 5.2 					(Subcompositional regression)			





