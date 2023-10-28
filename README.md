# Projects

## Outlier Statistic Methods:

The outlier sum statistic is a statistical method to identify differential features between 2 sets of data based on differences in the outliers in the 2 cohorts. Mostly, this sort of 
difference in outliers is observed in complex biological diseases (for example cancers) - where outlier individuals for genes potentially associated with the disease will be found. It
was first introduced by tibshiranine in the paper \href{https://doi.org/10.1093/biostatistics/kxl005}

## Calculation:
1. For a matrix of N x p, with N individuals and P features. Let N be split into N1 (set 1) and N2 (set 2)
2. For each feature, define a threshold based on the inter-quartile range (IQR) and tukey's fences as
   a. IQR_threshold = q75 + k x (q75 - q25)
   b. k = 1.0 for relaxed, 3.0 for conservative
3. Using the IQR threshold, find all samples for feature K \in R^{P} which are above the threshold
4. Calculate the Sum of values for all those samples - Let's call that OS_set_1
5. Similarly, perform permutations to find the sum of values in set 2
6. Define Z-Score for set 1 for each feature with respect to the permuted distribution.
7. Calcualte p-values for each feature to assign a feature importance value.


## Code Running:
1. There exists a demo notebook which can be used to run Outlier sum statistics on 2 pseudo data sets generated from normal distributions. 
