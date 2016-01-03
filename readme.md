Supplement for "Cascaded High Dimensional Histograms: A Generative Approach to Density Estimation"
=================================================================================================

We consider the problem of interpretable density estimation for high dimensional categorical data. In one or two dimensions, we would naturally consider histograms (bar charts) for simple density estimation problems. However, histograms do not scale to higher dimensions in an interpretable way, and one cannot usually visualize a high dimensional histogram. This repository presents implementation of two of the models presented in the paper ``Cascaded High Dimensional Histogram: A Generative Approach to Density Estimation'' to compute density trees.  The first one allows the user to specify the number of desired leaves in the tree as a Bayesian prior.  The second model allows the user to specify the desired number of rules and the length of rules within the prior and returns a list. 

Density List
_______________

The first model, densitylistmloss.py returns a list, a one sided tree, rule lists are easier to optimize than trees. Each tree can be expressed as a rule list, however, some trees may be more complicated to express as a rule list. By using lists, we implicitly hypothesize that the full space of trees may not be necessary and that simpler rule lists may suffice.

Leaf-based Cascade Model
__________________________

The second model, leafbasedmloss.py returns a general tree. We do not restrict ourselves to binary trees. Density at each leaf is reported. The main prior on tree *T* is on the number of leaf. We desired our trees to be interpretable besides being highly representative of the data.

To run the algorithms
_____________________
We provided the Python implementation of the code. One just have to call the function topscript in the respective class.
Two files have to be passed as the input to the model. Both files have to be TSV files.
Furthermore, we require for Density List algorithm, the categories names have to be distinct for different feature. If ``Male'' is used in feature 1, it can no longer be used in feature 2.

The file demomloss illustrates how can we run the topscript code. 
