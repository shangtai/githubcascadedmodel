import densitylistmloss
import leafbasedmloss

thefilename = 'stringdata'

#the following line demonstrate how to call the density list model
print "calling density list model"
permsdic, d_star, itemsets, theta, ci_theta, likelihood_d_star = densitylistmloss.topscript(thefilename)

#the following line illustrate how to call the leaf based density cascade model
print "calling leaf based density model"
leafObjectiveFunction = leafbasedmloss.topscript(thefilename)






