# ASCA

The ASCA algorithm for python, is based on the algorithm ANOVA - Simultaneous Component Analysis algorithm [[1]](#1) [[2]](#2).



ANOVA-Simultaneous Component Analysis:
    
ASCA partitions response matrix, Y, based on linear predictors in a design 
matrix X.
The structure of the study design is considered in the multivariate model, 
and the effect of indiviual factors can be analysed.
The responses are partitioned by Ordinary Least-Squares, and component
analysis is performed on the he effect matrices and the residual error 
term ε

Parameters
----------
**X** : *numpy ndarray or pandas Series/DataFrame*

Contain linear predictors so that:

$Y = β_0 + β_1X_1 + ... + β_jX_j + ε = Xβ + ε$

**Y**: *numpy ndarray or pandas DataFrame*

Matrix with response variables Y
    
Options
-------
**n_components**: *int*

Number of components in component analysis. Default is 5

**interaction**: *boolean*

Assumes interaction effect between responses. Default is false.
    
**n_perm**: *int*

Number of permutations for the permutation test. Default is 1000

Attributes
----------
*All attributes are stored in  ._results as pseudo private attributes. All
attributes are empty (NoneType) untill .fit() is called.*

**beta**: *numpy ndarray*

Beta (β) for all response variables
    
**column_names**: *list*

List of column names for response variables, from pandas.columns. If 
none are provided, a list with '*Variable i*' is created, where *i* is the 
column number
    
**contrast_matrix**: *list*

List of contrast matrices, based on experimental factors
    
**dummy**: *numpy ndarray*

Dummy matrix of the linear predictor X.

Example: 

Matrix with 6 observations of 3 factors with 2 and 3 levels:

    
    [[Foo, 1, n30],
     [Foo, 2, n20],
     [Bar, 1, n10],
     [Bar, 2, n30],
     [Baz, 1, n20],
     [Baz, 2, n10]]

returns:   
    
    [[ 1.,  0.,  1., -1.,  0.,  1.],
     [ 1.,  0.,  1.,  1.,  1.,  0.],
     [ 1., -1., -1., -1., -1., -1.],
     [ 1., -1., -1.,  1.,  0.,  1.],
     [ 1.,  1.,  0., -1.,  1.,  0.],
     [ 1.,  1.,  0.,  1., -1., -1.]]
     
    
and if ```.Options.interaction == True```:

    [[ 1.,  0.,  1., -1.,  0.,  1., -0., -1.,  0.,  1., -0., -1.],
     [ 1.,  0.,  1.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
     [ 1., -1., -1., -1., -1., -1.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1., -1., -1.,  1.,  0.,  1., -1., -1., -0., -1.,  0.,  1.],
     [ 1.,  1.,  0., -1.,  1.,  0., -1., -0.,  1.,  0., -1., -0.],
     [ 1.,  1.,  0.,  1., -1., -1.,  1.,  0., -1., -0., -1., -1.]]
   
**dummy_indexer**: *list*

List of boolean arrays to index dummy variables.
    
**dummy_n**: *list*

List of indexed dummy variables.
    
**factor_names**: *list*

List of factor names. If none are provided, a list with "Factor i" is created, where i is the factor number

**factors**: *int*

number of factors

**mu**: *numpy ndarray*

Expected value of the response:
$µ = Y - ε$

**mu_n**: *list*

List of partitioned effect matrices
    
**p_value**: *list*

List of p-values for each factor. A permutation test is used to calculate
the p-values.

**permutation_sse**: *numpy ndarray*

Sum of squares for each permuted models, where rows, n, are number of 
permutations and columns, m, are factors
    
**residuals**: *Numpy ndarray*

Matrix containing residuals

**sca_results**: *list*

List of SCA results. Each of the results contain scores, loadings, 
explained variance and the factor name, for each factor.
 
Methods
--------    
***fit()***:

Wrapper function. Performs algorithm
Results are stored in ._results (pseudo private attributes)
    
***permutation_test()***:

Calculates Sum of Squared Error for n models with each factor permuted
consecutively, where *n* is number of permutations (*.Options.n_perm*). 
The permutation of the factor is random and for the design matrix
    
    [[Foo, 1, n30],
     [Foo, 2, n20],
     [Bar, 1, n10],
     [Bar, 2, n30],
     [Baz, 1, n20],
     [Baz, 2, n10]]
    
The permutation of the first factor could be
    
    [[Baz, 1, n30],
     [Bar, 2, n20],
     [Foo, 1, n10],
     [Baz, 2, n30],
     [Foo, 1, n20],
     [Bar, 2, n10]]
    
The p-value is the number of SSE of the full model greater than the SSE of the
permuted model, divided by number of permutations: 
$\frac{\sum (SSEp < SSEf) + 1}{ n_{perm} + 1}$

*.fit()* is called when *permutation_test()* is called.
        
***plot_permutations():***

Plots histogram of SSE of the permuted models. SSE of the full model is 
marked in the histogram.
    
***plot_loadings():***

Plots loadings of selected components for the selected factor. Default 
is component 1 and 2 and factor 0.
    
***plot_scores():***

Plots scores of selected components for the selected factor. Default 
is component 1 and 2 and factor 0. Scores are grouped by supplied
vector (pandas Series), or a string with the name of a factor. If none 
are provided, the scores are grouped by the plotted factor.

***plot_raw():***

Plots raw data. Can be coloured according to individual factors.
    
***plot_raw():***

Plots residuals. Can be coloured according to individual factors.


## References
<a id="1">[1]</a> 
Smilde, Age K. et al. “ANOVA-Simultaneous Component Analysis (ASCA): a New Tool for Analyzing Designed Metabolomics Data.” Bioinformatics 21.13 (2005): 3043–3048. Web.

<a id="2">[2]</a> 
Jansen, Jeroen J. et al. “ASCA: Analysis of Multivariate Data Obtained from an Experimental Design.” Journal of chemometrics 19.9 (2005): 469–481. Web.
