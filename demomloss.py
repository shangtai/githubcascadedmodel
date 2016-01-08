import densitylistmloss
import leafbasedmloss
from numpy import *

thefilename = 'stringdata'

#the following line demonstrate how to call the density list model
print "calling density list model"
#permsdic, d_star, itemsets, theta, ci_theta, likelihood_d_star = densitylistmloss.topscript(thefilename,3,1,array([1.])) # or
permsdic, d_star, itemsets, theta, ci_theta, likelihood_d_star = densitylistmloss.topscript(thefilename,3,1)
#besides the filename, the other parameters are optional
#the second parameter is the desired length of the list
#the third parameter is the desired width of the rule
#the fourth parameter is the alpha, an array for dirichlet distribution

#the following line illustrate how to call the leaf based density cascade model
print "calling leaf based density model"
leafObjectiveFunction = leafbasedmloss.topscript(thefilename,[3],2)
#besides the filename, the other parameters are optional
#the second parameter has to be a list, it strores the desired lengths, we will perform cross validation
#the third parameter is the alpha, a parameter for dirichlet distribution






