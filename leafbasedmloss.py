#######Supplement for "Cascaded High Dimensional Histograms: A Generative Approach to Density Estimation."

###LICENSE
#
#This software is released under the MIT license.
#
#Copyright (c) 2016-17 Siong Thye Goh & Cynthia Rudin
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
#
#The author/copyright holder can be contacted at bletham@mit.edu

####README
#
#This code implements the Cascaded Leaf-based Density tree algorithm as described in the 
#paper. We include a simulated dataset in the correct formatting to be used
#by this code.
#
#
# ##INPUT
#
#The main code to be run by the user is the function "topscript". In addition to 
#the parameters described in the paper, in this function you must specify a 
#variable "fname" that will specify which data to load.
#The following files must be present and must be formatted as described (see 
#titanic data included for an example):
#
# - fname_train.tab : This is the file containing the training X data, for which 
#all features are strings. Each line is a data entry in which all of the features 
#with value "1" are simply listed, with spaces delimiting different features. For 
#example, in the Titanic dataset, "1st_class adult male" is a line in 
#titanic.tab. Furthermore, each feature category name has to be distinct.
#
#
# - fname_test.tab formatted with the same formatting as the 
#training data.
#
# ##OUTPUT
#
#The highest level function, "topscript," returns:
#
# - A tree- This is the tree that we found, a tree is a list of nodes. Node stores information such as who are the parents, children, ancestor and descendant of itself. 
# - likelihood - Likelihood value for the test data
#
# In the print out, we can see each leaf and description of the leaf in a list of list, where the first component is the feature index and the second component is the description of the categories of the features.

#The algorithm uses a simulated annealing approach to find the tree.

# ##CODE:


import random
import csv
import copy
import math
import time
from collections import defaultdict
from operator import itemgetter
from numpy import prod
import numpy as np
import pylab 
import matplotlib.pyplot as plt
import pandas as pd
import itertools

poissonLambda=2
flagcount=0
margin=0.01
smallDensity=0.001


def topscript(thefilename,priorparameter=[3],localalpha=2):
    global featureTypeVector, alpha
    alpha=localalpha
    data=load_data(thefilename+'_train') #preparing training data
    testdata=load_data(thefilename+'_test') #preparing test data
    featureTypeVector=['cat']*len(data) #the original code was meant for something bigger, hence this akward line
    varietyDetailMatrix = featureVarietyDetail(data)
    varietyMatrix = featureVariety(data)
    D=len(data[0])
    repeatingindex=0
    leafObjectiveFunction = simulatedannealingcrossvalidation(data,testdata, varietyMatrix,D,featureTypeVector,varietyDetailMatrix,[],[],priorparameter, False, repeatingindex,False)
    print "the likelihood for test data is "+str(leafObjectiveFunction[1])

def logDirichletNormalizingDiff(alpha,countVector):
    fakeCountVector=[c+alpha for c in countVector]
    return sum([logfactorialdiff(alpha-1, fakeCount) for fakeCount in fakeCountVector])-logfactorialdiff(len(countVector)*alpha-2, len(countVector)*alpha+sum(countVector))

def featureVariety(dataset):
    global featureTypeVector
    datasetTranpose = [[row[i] for row in dataset] for i in range(len(dataset[0]))]
    fV=[]
    for i in range(len(datasetTranpose)):
        if featureTypeVector[i]=='con': 
            fV.append(max(datasetTranpose[i])-min(datasetTranpose[i]))
        elif featureTypeVector[i]=='cat':
            fV.append(len(set(datasetTranpose[i])))
        else:
            fV.append(len(featureTypeVector[i]))
    return fV
    
def featureVarietyDetail(dataset):
    global featureTypeVector
    datasetTranpose = [[row[i] for row in dataset] for i in range(len(dataset[0]))]
    fVD=[]
    for i in range(len(datasetTranpose)):
        if featureTypeVector[i]== 'con':
            fVD.append([min(datasetTranpose[i]),max(datasetTranpose[i])])
        elif featureTypeVector[i] == 'cat':
            fVD.append(list(set(datasetTranpose[i])))
        else:
            tmp=dict()
            for j in range(len(featureTypeVector[i])):
                tmp[featureTypeVector[i][j]]=j
            fVD.append(tmp)
    return fVD
    
def mean(inputlist):
    return sum(inputlist)/float(len(inputlist))

def std(inputlist):
    mymean=mean(inputlist)
    return math.sqrt(mean([(x-mymean)**2 for x in inputlist]))

def logfactorialdiff(a,b):
    return sum(map(math.log,range(a+1,b+1)))
    
def logfact(x):
    return sum([math.log(x) for x in range(1,x+1)])
   
def diff(a,b):
    #compute difference of two lists, a and b
    c=set(b)
    return [aa for aa in a if aa not in c]
    
def euclidean(p,q):
    return math.sqrt(sum([(p[i]-q[i])**2 for i in range(len(p))]))

def hellinger(p,q):
    return math.fabs(2-(2.0/len(p))*sum([ math.sqrt(q[i]/p[i]) for i in range(len(p))]))
    #return 2-(2.0/len(p))*sum([ math.sqrt(q[i]*p[i]) for i in range(len(p))])
    #return math.sqrt(sum([(math.sqrt(p[i])-math.sqrt(q[i]))**2 for i in range(len(p))]))/math.sqrt(2)
 
def likelihood(p):
    global smallDensity
    q=p[:]
    answer=0
    for i in q:
        if i<smallDensity:
            answer+=math.log(smallDensity)
        else:
            answer+=math.log(i)
    return answer

probabilityvector=[0.99/4,0.99*0.5, 0.99*0.75,0.99]

maxm=100

def logpriorfunction(k, priorlambda):
    return -priorlambda+k*math.log(priorlambda)-sum([math.log(i) for i in range(1,k+1)])
    
class Node():
    def __init__(self,num,leaf):
        self.me=num  
        self.parent= -1
        self.child=[]
        self.ancestor=[]  #list recording the ancestor number
        self.descendant=[]
        self.feature= -1  # negative number indicates that we have not started doing anything
        self.descendantfeature=[]  
        self.ancestorfeaturewithbinary=[]
        self.active=False #to know if the number is used, to avoid the usage of big index unnecessarily.
        self.leaf=leaf
        self.subdata=[]
        self.count=0  
    
    def getDepth(self):
        return len(self.ancestorfeaturewithbinary)
        
    def getNotUsedComplexity(self):
        if self.ancestorfeaturewithbinary == []:
            featureNotUsed = range(len(data[0]))
            parentVolume = 1
        else:
            featureNotUsed = diff(range(len(data[0])),[featureUsed[0] for featureUsed in self.ancestorfeaturewithbinary])
            #parentVolume = [featureUsed[1][1]-featureUsed[1][0] for featureUsed in self.ancestorfeaturewithbinary if featureTypeVector[featureUsed[0]]!='cat']
            parentVolume = []
            parentVolume.extend([featureUsed[1][1]-featureUsed[1][0] for featureUsed in self.ancestorfeaturewithbinary if featureTypeVector[featureUsed[0]]!='cat'])
            #parentVolume.extend([len(featureUsed[1]) for featureUsed in self.ancestorfeaturewithbinary if featureTypeVector[featureUsed[0]]=='cat'])
            featureUsedVector = list(set([featureUsed[0] for featureUsed in self.ancestorfeaturewithbinary]))
            for featureUsed in featureUsedVector:
                parentVolume.append(len([subcategories[1] for subcategories in self.ancestorfeaturewithbinary if subcategories[0]==featureUsed][-1]))
        return prod([varietyMatrix[i] for i in featureNotUsed])*prod(parentVolume)

       
    def computeDensity(self):
        if len(self.subdata)==0: # and len(self.testsubdata)>0:
            return 0.0#10**(-6)/(float(ndata)*self.getNotUsedComplexity())
        else: 
            return len(self.subdata)/(float(ndata)*self.getNotUsedComplexity())
        
    def addoffspring(self,child,feature):
        #child is a list of numbers and the corresponding features used to split
        self.child=child
        self.feature=feature
        self.descendant=child
    
    def addparent(self,parent):
        self.parent=parent

    def updateancestor(self,latestancestorlist,latestancestorfeaturewithbinary):
        self.ancestor=latestancestorlist[:]#copy.deepcopy(latestancestorlist)
        self.ancestorfeaturewithbinary= latestancestorfeaturewithbinary[:] #copy.deepcopy(latestancestorfeaturewithbinary)
        
    def updatedescendant(self,latestdescendantlist,latestdescendantfeatures):
        self.descendant=latestdescendantlist[:]#copy.deepcopy(latestdescendantlist)
        self.descendantfeature=latestdescendantfeatures[:]#copy.deepcopy(latestdescendantfeatures)

    def crossvalidationObjective(self):
        nodeSize = len(self.subdata)
        return (nodeSize/float(ndata)-2*(nodeSize-1)/float(ndata-1))*self.computeDensity()

    def crossvalidationFixedPart(self):
        return (len(self.subdata)/float(ndata))*self.computeDensity()

    def getChild(self):
        return self.child

    def getNumberOfBranch(self):
        return len(self.child)

#a tree is a collection of nodes
class Tree():
    def __init__(self,D,data):
        leafsize=1
        self.tree=[]
        #self.objective=1
        self.tree.append(Node(0,True))
        self.tree[0].active = True
        self.maxindex=1
        self.tree[0].subdata=range(len(data))
        #print "end of initialization"

    def warmStart(self,D,data,detTreeFile):
        usedNumbers=[0]
        leafVisitation=[]
        parallelNodes=[0]
        self.tree[0].leaf=False
        with open(detTreeFile,'rb') as readTreeFile:
            treeFileReader=csv.reader(readTreeFile,delimiter=',')
            next(treeFileReader,None)
            for row in treeFileReader:
                temprow = [x for x in str.split(row[0]) if x !='Var.' and x != '|' and x!=':']
                if len(temprow)<4: #this is not a leaf, record variable, the boundary with sign and the float and the value 0
                    leafVisitation.append([int(temprow[0]),temprow[1],float(temprow[2]),0])
                    nextNumber=max(usedNumbers)+1
                    usedNumbers.append(nextNumber)
                    parallelNodes.append(nextNumber)
                    self.tree.append(Node(nextNumber, False))
                    self.tree[parallelNodes[-1]].active= True
                    feature = leafVisitation[-1][0]
                    self.tree[parallelNodes[-2]].feature = feature
                    self.tree[parallelNodes[-1]].parent = parallelNodes[-2]
                    usedCategoriesForFeatures=[self.tree[parallelNodes[-2]].ancestorfeaturewithbinary[i][1] for i in xrange(len(self.tree[parallelNodes[-2]].ancestorfeaturewithbinary)) if self.tree[parallelNodes[-2]].ancestorfeaturewithbinary[i][0]==feature]
                    if len(usedCategoriesForFeatures)>0:
                        usedCategoriesForFeatures=usedCategoriesForFeatures[-1] # the last part should be the smallest
                    else:
                        usedCategoriesForFeatures=varietyDetailMatrix[feature][:]
                    if leafVisitation[-1][1]=="<=":
                        tempVarietyFeatureDetail=[pseudoCategories for pseudoCategories in usedCategoriesForFeatures if int(pseudoCategories)<= leafVisitation[-1][2]]
                    else:
                        tempVarietyFeatureDetail=[pseudoCategories for pseudoCategories in usedCategoriesForFeatures if int(pseudoCategories) > leafVisitation[-1][2]]
                    self.tree[parallelNodes[-1]].updateancestor(self.tree[parallelNodes[-2]].ancestor+[parallelNodes[-2]],self.tree[parallelNodes[-2]].ancestorfeaturewithbinary+[[feature,tempVarietyFeatureDetail]])
                    self.tree[parallelNodes[-1]].subdata=[ x for x in self.tree[parallelNodes[-2]].subdata if data[x][feature] in tempVarietyFeatureDetail ]

                    #self.tree[node].addoffspring(tmpnotactive,feature)
                    self.tree[parallelNodes[-2]].child.append(parallelNodes[-1])
                    self.tree[parallelNodes[-2]].feature=feature
                    self.tree[parallelNodes[-2]].descendant.append(parallelNodes[-1])
                    for i in self.tree[parallelNodes[-2]].ancestor:
                        self.tree[i].updatedescendant(self.tree[i].descendant+[parallelNodes[-1]],list(set(self.tree[i].descendantfeature).union(set([feature]))))
                    #add child here
                else:
                    #now I am inside a leaf, do something to conclude and backtrack
                    #detSubstructure=[]
                    leafVisitation.append([int(temprow[0]),temprow[1],float(str.split(temprow[2],':')[0])])
                    #
                    nextNumber=max(usedNumbers)+1
                    usedNumbers.append(nextNumber)
                    parallelNodes.append(nextNumber)
                    self.tree.append(Node(nextNumber, True))#yup, that is a leaf
                    self.tree[parallelNodes[-1]].active= True 
                    feature = leafVisitation[-1][0]
                    self.tree[parallelNodes[-2]].feature = feature
                    self.tree[parallelNodes[-1]].parent = parallelNodes[-2]
                    usedCategoriesForFeatures=[self.tree[parallelNodes[-2]].ancestorfeaturewithbinary[i][1] for i in xrange(len(self.tree[parallelNodes[-2]].ancestorfeaturewithbinary)) if self.tree[parallelNodes[-2]].ancestorfeaturewithbinary[i][0]==feature]
                    if len(usedCategoriesForFeatures)>0:
                        usedCategoriesForFeatures=usedCategoriesForFeatures[-1] # the last part should be the smallest
                    else:
                        usedCategoriesForFeatures=varietyDetailMatrix[int(feature)][:]
                    if leafVisitation[-1][1]=="<=":
                        tempVarietyFeatureDetail=[pseudoCategories for pseudoCategories in usedCategoriesForFeatures if int(pseudoCategories)<= leafVisitation[-1][2]]
                    else:
                        tempVarietyFeatureDetail=[pseudoCategories for pseudoCategories in usedCategoriesForFeatures if int(pseudoCategories) > leafVisitation[-1][2]]
                    self.tree[parallelNodes[-1]].updateancestor(self.tree[parallelNodes[-2]].ancestor+[parallelNodes[-2]],self.tree[parallelNodes[-2]].ancestorfeaturewithbinary+[[feature,tempVarietyFeatureDetail]])
                    self.tree[parallelNodes[-1]].subdata=[ x for x in self.tree[parallelNodes[-2]].subdata if data[x][feature] in tempVarietyFeatureDetail ]

                    #self.tree[node].addoffspring(tmpnotactive,feature)
                    self.tree[parallelNodes[-2]].child.append(parallelNodes[-1])
                    self.tree[parallelNodes[-2]].feature=feature
                    self.tree[parallelNodes[-2]].descendant.append(parallelNodes[-1])
                    for i in self.tree[parallelNodes[-2]].ancestor:
                        self.tree[i].updatedescendant(self.tree[i].descendant+[parallelNodes[-1]],list(set(self.tree[i].descendantfeature).union(set([feature]))))
                    #fixedPart+=leafVolume*float(str.split(temprow[3],'=')[1])**2 #update fixedPart
                    if len(leafVisitation)>1:
                        leafVisitation[len(leafVisitation)-2][3]+=1
                    leafVisitation.pop()
                    parallelNodes.pop()
                    if len(leafVisitation)>0:
                        while (leafVisitation[-1][3]==2):
                            if len(leafVisitation)>1:
                                leafVisitation[-2][3]+=1
                            leafVisitation.pop()
                            parallelNodes.pop()
                            if leafVisitation==[]:
                                break
            self.maxindex=len(usedNumbers) #remember to handle trivial case
            
    def __str__(self): #return tree size for the time being.
        global ndata
        returnstring=""
        for node in self.tree:
            if node.leaf==True and node.active==True:
                returnstring+=str(node.me)
                returnstring+=':'
                returnstring+=str(node.ancestorfeaturewithbinary)
                returnstring+='~'
                returnstring+='\n'
                ni=0
                for i in range(ndata):
                    countflag=1
                    for j in range(len(node.ancestorfeaturewithbinary)):
                        if data[i][node.ancestorfeaturewithbinary[j][0]] != node.ancestorfeaturewithbinary[j][1]:
                            countflag=0
                            break
                    if countflag == 1:
                        ni+=1
                returnstring+=str(ni)
                returnstring+='\n'
        return returnstring
        
    def datadistribution(self,data):
        global ndata
        returnstring=""
        counter=0
        for node in self.tree:
            if node.leaf==True and node.active==True:
                returnstring+='leaf '+str(counter)#str(node.me)
                returnstring+=':'
                returnstring+=str(node.ancestorfeaturewithbinary)
                returnstring+='~'
                returnstring+='\n'
                ni=len(node.subdata)
                returnstring+="density of the node is "+str(node.computeDensity())
                #returnstring+=str(ni)
                returnstring+='\n'
                returnstring+='\n\n'
                counter+=1
        return returnstring
    
    def getValidLeaves(self):
        return [node for node in self.tree if node.leaf  and node.active]

    def getValidNodes(self):
        return [node for node in self.tree if node.active]

    def getNumberOfFeaturesUsed(self):
        leafNodes = self.getValidLeaves()
        featuresUsed = []
        for leaf in leafNodes:
            for ancestorlist in leaf.ancestorfeaturewithbinary:
                featuresUsed.append(ancestorlist[0])
        featuresUsed=set(featuresUsed)
        return len(featuresUsed)

    def crossvalidationObjective(self,data):
        return sum([node.crossvalidationObjective() for node in self.getValidLeaves()])

    def crossvalidationForTest(self,testdata):
        return sum([node.crossvalidationFixedPart() for node in self.getValidLeaves()])-2*sum(self.getDensityForTest(testdata))/float(len(testdata))

    def getBelongingForAPoint(self,x):
        global featureTypeVector
        #print featureTypeVector
        validLeaves = self.getValidLeaves()
        if len(validLeaves)==1:
            return validLeaves[0].me
        foundFlag=False
        for validLeaf in validLeaves:
            exploreSoFar=0
            exploreDepth=len(validLeaf.ancestorfeaturewithbinary)
            #print "exploreDepth",exploreDepth
            for feature, detail in validLeaf.ancestorfeaturewithbinary:
                if featureTypeVector[feature] == 'cat':
                    if x[feature] not in detail:
                        break
                elif featureTypeVector[feature] == 'con':
                    if len(detail)==2:
                        if x[feature]>detail[1] or x[feature]<detail[0]:
                            break
                    else:
                        if detail[2]=='first':
                            if x[feature]>detail[1]:
                                break
                        else:
                            if x[feature]<detail[0]:
                                break
                else:
                    if featureVarietyDetail[feature][x[feature]]>detail[1] or featureVarietyDetail[feature][x[feature]]<detail[0]:
                        break
                exploreSoFar += 1
                #print "exploreSoFar", exploreSoFar
            if exploreSoFar == exploreDepth:
                return validLeaf.me

    def getDensityForTest(self,testdata):
        belongingVector = [self.getBelongingForAPoint(x) for x in testdata]
        validLeaves = self.getValidLeaves()
        indexLeaves = [ validLeaf.me for validLeaf in validLeaves]
        densityLeaves = [ validleaf.computeDensity() for validleaf in validLeaves]
        leavessize =[len(validleaf.subdata) for validleaf in validLeaves]
        return [densityLeaves[indexLeaves.index(i)] for i in belongingVector]



    def removenode(self,node):
        #remove everything below that node, keep the node though.
        #note that the node here cannot be a leaf, otherwise, there is nothing to remove.
        #reset the features of descendant of features at the end.
        for i in self.tree[node].descendant:
            self.tree[i]=Node(i,False) #reset, note that they are not active, not sure if we will encounter memory issue here.
            self.tree[i].active=False
            self.tree[i].subdata=[]
        for i in self.tree[node].ancestor:
            self.tree[i].updatedescendant(diff(self.tree[i].descendant,self.tree[node].descendant),diff(self.tree[i].descendantfeature,self.tree[node].descendantfeature))
        self.tree[node].descendant=[]
        self.tree[node].child=[]
        self.tree[node].feature=-1
        self.tree[node].leaf= True

    def addnode(self,node,feature):
        #to add 2 children to the node which was a leaf
        #first I need to find indexes that are not active
        #this code is embarrassing... but I will improve it later.
        #a remark is whether the feature should be chosen randomly
        global featureTypeVector, flagcount
 
        tmpnotactive=[]  #can be improved
        goAhead = True

        if featureTypeVector[feature] == 'con':
            numberOfChildren=np.random.poisson(poissonLambda,1).tolist()[0]+2 #make sure at least 2 children
            normalizedSplittingPoints=np.random.dirichlet([1]*numberOfChildren,1)
            normalizedSplittingPoints=normalizedSplittingPoints.tolist()[0]
            lendirichletsample=len(normalizedSplittingPoints)
            for dirichletindex in range(1,lendirichletsample):
                normalizedSplittingPoints[dirichletindex]+=normalizedSplittingPoints[dirichletindex-1]
            normalizedSplittingPoints=[0]+normalizedSplittingPoints
            splittingPoints=[varietyDetailMatrix[feature][0]-margin+theProportion*(varietyDetailMatrix[feature][1]+2*margin-varietyDetailMatrix[feature][0]) for theProportion in normalizedSplittingPoints]
        elif featureTypeVector[feature] == "cat":
            usedCategoriesForFeatures=[self.tree[node].ancestorfeaturewithbinary[i][1] for i in xrange(len(self.tree[node].ancestorfeaturewithbinary)) if self.tree[node].ancestorfeaturewithbinary[i][0]==feature]
            if len(usedCategoriesForFeatures)>0:
                usedCategoriesForFeatures=usedCategoriesForFeatures[-1] # the last part should be the smallest
            else:
                usedCategoriesForFeatures=varietyDetailMatrix[feature][:]
            numberOfChildren=len(usedCategoriesForFeatures)
        else:
            aRandomNumber=random.randint(0,2**(varietyMatrix[feature]-1)-1)
            flagcount+=1
            binaryLength = len(str(bin(aRandomNumber)))-2
            splittingPoints=[i for i in range(binaryLength) if (aRandomNumber>>i)&1 == 1]
            numberOfChildren = len(splittingPoints)+1
            splittingPoints=[-1]+splittingPoints+[varietyMatrix[feature]-1]
        for i in range(self.maxindex):
            if self.tree[i].active == False:
                tmpnotactive.append(i)
                if len(tmpnotactive) == numberOfChildren:
                    break
        tmpLength=len(tmpnotactive)
        if len(tmpnotactive) < numberOfChildren:
            for i in range(numberOfChildren-tmpLength):
                tmpnotactive.append(self.maxindex+i)
                self.tree.append(Node(tmpnotactive[-1],True))
            self.maxindex+=numberOfChildren-tmpLength
        

        if len(self.tree[node].subdata)==0 or numberOfChildren<2:
            goAhead = False

        if goAhead:
            self.tree[node].leaf=False
            self.tree[node].feature=feature
            for i in range(numberOfChildren):
                self.tree[tmpnotactive[i]].active = True
                self.tree[tmpnotactive[i]].parent = node
                self.tree[tmpnotactive[i]].leaf = True
                if featureTypeVector[feature] == 'cat':
                    self.tree[tmpnotactive[i]].updateancestor(self.tree[node].ancestor+[node],self.tree[node].ancestorfeaturewithbinary+[[feature,[usedCategoriesForFeatures[i]]]])
                    self.tree[tmpnotactive[i]].subdata=[ x for x in self.tree[node].subdata if data[x][self.tree[node].feature] == usedCategoriesForFeatures[i] ]
                elif featureTypeVector[feature] == 'con':
                    if i==0:
                        self.tree[tmpnotactive[i]].updateancestor(self.tree[node].ancestor+[node],self.tree[node].ancestorfeaturewithbinary+[[feature,[splittingPoints[i],splittingPoints[i+1], 'first']]])
                    elif i==numberOfChildren-1:
                        self.tree[tmpnotactive[i]].updateancestor(self.tree[node].ancestor+[node],self.tree[node].ancestorfeaturewithbinary+[[feature,[splittingPoints[i],splittingPoints[i+1],'last']]])
                    else:
                        self.tree[tmpnotactive[i]].updateancestor(self.tree[node].ancestor+[node],self.tree[node].ancestorfeaturewithbinary+[[feature,[splittingPoints[i],splittingPoints[i+1]]]])
                    self.tree[tmpnotactive[i]].subdata= [ x for x in self.tree[node].subdata if (data[x][self.tree[node].feature] >= splittingPoints[i] and data[x][self.tree[node].feature]< splittingPoints[i+1])]
                else:
                    self.tree[tmpnotactive[i]].updateancestor(self.tree[node].ancestor+[node],self.tree[node].ancestorfeaturewithbinary+[[feature,[splittingPoints[i],splittingPoints[i+1]]]])
                    self.tree[tmpnotactive[i]].subdata= [ x for x in self.tree[node].subdata if (varietyDetailMatrix[feature][data[x][self.tree[node].feature]] > splittingPoints[i] and  varietyDetailMatrix[feature][data[x][self.tree[node].feature]] <= splittingPoints[i+1])]
            self.tree[node].addoffspring(tmpnotactive,feature)
        
            for i in self.tree[node].ancestor:
                self.tree[i].updatedescendant(self.tree[i].descendant+self.tree[node].child,list(set(self.tree[i].descendantfeature).union(set([feature]))))


    def splitnode(self,node):
        #print "splitting"
        parentNode = self.tree[self.tree[node].parent]
        parentNodeFeature = parentNode.feature
        
        tmpnotactive=[]
        for i in range(self.maxindex):
            if self.tree[i].active == False:
                tmpnotactive.append(i)
                if len(tmpnotactive) == 1:
                    break
        tmpLength=len(tmpnotactive)
        if len(tmpnotactive) < 1:
            for i in range(1-tmpLength):
                tmpnotactive.append(self.maxindex+i)
                self.tree.append(Node(tmpnotactive[-1],True))
            self.maxindex+=1

        goAhead = True
        if featureTypeVector[parentNodeFeature] == 'con':
            leftPoint = self.tree[node].ancestorfeaturewithbinary[-1][1][0]
            rightPoint = self.tree[node].ancestorfeaturewithbinary[-1][1][1]
            middlePoint = random.uniform(leftPoint, rightPoint)
            leftSubdata = [x for x in self.tree[node].subdata if data[x][parentNodeFeature]< middlePoint]
            rightSubdata = [x for x in self.tree[node].subdata if data[x][parentNodeFeature] >= middlePoint]
            if len(leftSubdata)==0 or len(rightSubdata) == 0:
                goAhead=False
        elif featureTypeVector[parentNodeFeature]== 'cat':
            groupedCategories=[x[1] for x in self.tree[node].ancestorfeaturewithbinary if x[0]==parentNodeFeature]
            #try:
            groupedCategories=groupedCategories[-1]
            #except:
            #print groupedCategories, node, self.tree[node].ancestorfeaturewithbinary, parentNodeFeature, self.tree[node].parent, "grouped, node, ancestorfeature, parentNodeFeature"
            #print self.datadistribution(data)
            if len(groupedCategories)==1:
                self.tree[tmpnotactive[0]].active = False
                goAhead=False
            else:
                lengthOfFirstPart = random.randint(1, len(groupedCategories)-1)
                firstPart = random.sample(groupedCategories, lengthOfFirstPart)
                secondPart = diff(groupedCategories, firstPart)
                leftSubdata =[x for x in self.tree[node].subdata if data[x][parentNodeFeature] in firstPart]
                rightSubdata = [x for x in self.tree[node].subdata if data[x][parentNodeFeature] in secondPart]
                if len(leftSubdata)==0 or len(rightSubdata)==0:
                    goAhead = False
        else:
            leftPoint = self.tree[node].ancestorfeaturewithbinary[-1][1][0]
            rightPoint = self.tree[node].ancestorfeaturewithbinary[-1][1][1]
            if rightPoint-leftPoint==1:
                self.tree[tmpnotactive[0]].active = False
                goAhead = False
            else:
                self.tree.removenode(node)
                middlePoint = random.randint(leftPoint+1, rightPoint-1)
                leftSubdata = [x for x in self.tree[node].subdata if data[x][parentNodeFeature] in range(leftPoint+1,middlePoint+1)]
                rightSubdata = diff(self.tree[node].subdata, leftSubdata)
                if len(leftSubdata) == 0 or len(rightSubdata) == 0:
                    goAhead = False


        if goAhead:
            self.tree[tmpnotactive[0]].parent = self.tree[node].parent
            self.tree[tmpnotactive[0]].active = True
            self.tree[tmpnotactive[0]].leaf = True
            if featureTypeVector[parentNodeFeature] == 'con':
                self.removenode(node)
                self.tree[self.tree[node].parent].child+=[tmpnotactive[0]]
                self.tree[node].subdata = leftSubdata
                self.tree[tmpnotactive[0]].subdata = rightSubdata
                self.tree[node].ancestorfeaturewithbinary [-1][1][0]=leftPoint
                self.tree[node].ancestorfeaturewithbinary[-1][1][1] = middlePoint
                self.tree[tmpnotactive[0]].updateancestor(parentNode.ancestor+[parentNode.me],parentNode.ancestorfeaturewithbinary+[[self.tree[node].ancestorfeaturewithbinary[-1][0],[middlePoint,rightPoint]]])
                self.tree[self.tree[node].parent].descendant +=[tmpnotactive[0]]
                for i in self.tree[self.tree[node].parent].ancestor:
                    self.tree[i].updatedescendant(self.tree[i].descendant+[tmpnotactive[0]],self.tree[i].descendantfeature)
                if len(self.tree[node].ancestorfeaturewithbinary[-1][1])==3:
                    if self.tree[node].ancestorfeaturewithbinary[-1][1][2] =='last':
                        self.tree[node].ancestorfeaturewithbinary[-1][1].pop()
                        self.tree[tmpnotactive[0]].ancestorfeaturewithbinary[-1][1].append('last')
            elif featureTypeVector[parentNodeFeature] == 'cat':
                groupedCategories=[x[1] for x in self.tree[node].ancestorfeaturewithbinary if x[0]==parentNodeFeature]
                if len(groupedCategories)==1:
                    self.tree[tmpnotactive[0]].active = False
                else:
                    self.removenode(node)
                    self.tree[node].subdata = leftSubdata
                    self.tree[self.tree[node].parent].child+=[tmpnotactive[0]]
                    self.tree[tmpnotactive[0]].subdata = rightSubdata
                    self.tree[node].ancestorfeaturewithbinary[-1][1]= firstPart
                    self.tree[tmpnotactive[0]].updateancestor(parentNode.ancestor+[parentNode.me],parentNode.ancestorfeaturewithbinary+[[self.tree[node].ancestorfeaturewithbinary[-1][0],secondPart]])
                    self.tree[self.tree[node].parent].descendant +=[tmpnotactive[0]]
                    for i in self.tree[self.tree[node].parent].ancestor:
                        self.tree[i].updatedescendant(self.tree[i].descendant+[tmpnotactive[0]],self.tree[i].descendantfeature)
            else: 
                leftPoint = self.tree[node].ancestorfeaturewithbinary[-1][1][0]
                rightPoint = self.tree[node].ancestorfeaturewithbinary[-1][1][1]
                if rightPoint-leftPoint==1:
                    self.tree[tmpnotactive[0]].active = False
                if self.tree[tmpnotactive[0]].active:
                    self.tree[self.tree[node].parent].addoffspring(self.tree[self.tree[node].parent].child+[tmpnotactive[0]], parentNodeFeature)
                    for i in self.tree[self.tree[node].parent].ancestor:
                        self.tree[i].updatedescendant(self.tree[i].descendant+[tmpnotactive[0]],self.tree[i].descendantfeature)
                    self.tree[tmpnotactive[0]].parent=self.tree[node].parent
                    self.tree[self.tree[node].parent].child+=[tmpnotactive[0]]
                    self.tree[node].ancestorfeaturewithbinary[-1][1][1]=middlePoint
                    self.tree[tmpnotactive[0]].updateancestor(parentNode.ancestor+[parentNode.me],parentNode.ancestorfeaturewithbinary+[[self.tree[node].ancestorfeaturewithbinary[-1][0],[middlePoint,rightPoint]]])

    def mergeNode(self,node1,node2):
        parentNode = self.tree[self.tree[node1].parent]
        parentNodeFeature = parentNode.feature
        #print "merging"

        if len(parentNode.child)>=3:
            self.removenode(node1)
            self.removenode(node2)
            self.tree[node1].subdata=self.tree[node1].subdata+self.tree[node2].subdata
            self.tree[node2].active = False

            if featureTypeVector[parentNodeFeature] == 'con':
                self.tree[node1].ancestorfeaturewithbinary[-1][1][0]=min([self.tree[node1].ancestorfeaturewithbinary[-1][1][0],self.tree[node2].ancestorfeaturewithbinary[-1][1][0]])
                self.tree[node1].ancestorfeaturewithbinary[-1][1][1]=max([self.tree[node1].ancestorfeaturewithbinary[-1][1][1],self.tree[node2].ancestorfeaturewithbinary[-1][1][1]])
                if len(self.tree[node2].ancestorfeaturewithbinary[-1][1])==3:
                    if self.tree[node2].ancestorfeaturewithbinary[-1][1][2]=='last':
                        self.tree[node1].ancestorfeaturewithbinary[-1][1].append('last')
            elif featureTypeVector[parentNodeFeature] == 'cat':
                self.tree[node1].ancestorfeaturewithbinary[-1][1] = self.tree[node1].ancestorfeaturewithbinary[-1][1]+self.tree[node2].ancestorfeaturewithbinary[-1][1]
            else:
                self.tree[node1].ancestorfeaturewithbinary[-1][1][0]=min([self.tree[node1].ancestorfeaturewithbinary[-1][1][0],self.tree[node2].ancestorfeaturewithbinary[-1][1][0]])
                self.tree[node1].ancestorfeaturewithbinary[-1][1][1]=max([self.tree[node1].ancestorfeaturewithbinary[-1][1][1],self.tree[node2].ancestorfeaturewithbinary[-1][1][1]])

            self.tree[self.tree[node1].parent].child=diff(self.tree[self.tree[node1].parent].child,[node2])
            
            for i in self.tree[node1].ancestor:
                self.tree[i].updatedescendant(diff(self.tree[i].descendant,[node2]),self.tree[i].descendantfeature)
    
    def computeHellinger(self,testdata):
        global ftestVector
        whatWeFound=self.getDensityForTest(testdata)
        #print "our smallest value", min(whatWeFound)
        #print "the truth smallest", min(ftestVector)
        return hellinger(ftestVector,whatWeFound)
    
    def computeHellingerTrain(self,traindata):
        global ftrainVector
        whatWeFound=self.getDensityForTest(traindata)
        return hellinger(ftrainVector,whatWeFound)


    def computeLikelihood(self,testdata):
        return likelihood(self.getDensityForTest(testdata))
    
    def objectivevalue(self,data, priorValue):
        global alpha
        ndata=len(data)
        countleaf=self.countleafoftree()
        factorialcomputation=0
        nodelist=self.getValidLeaves()
        nodecount=[0]*len(nodelist)
        notUsedComplexityVector=[node.getNotUsedComplexity() for node in nodelist]
        nodeindex=0
        for node in nodelist:
            nodecount[nodeindex]=len(node.subdata)
            nodeindex+=1
        loglikelihood=sum([logfactorialdiff(alpha-1,nodecount[i]+alpha-1)-nodecount[i]*math.log(notUsedComplexityVector[i]) for i in range(len(nodelist))])-logfactorialdiff(countleaf*alpha-1,len(data)+countleaf*alpha-1)
        return -loglikelihood-logpriorfunction(countleaf,priorValue)

    #def branchobjectivevalue(self,data, priorValue):
    #    ndata=len(data)
    #    dimensionsize=len(data[0])
    #    countleaf=self.countleafoftree()
    #    factorialcomputation=0
    #    leaflist=self.getValidLeaves()
    #    leafcount=[0]*len(leaflist)
    #    notUsedComplexityVector=[node.getNotUsedComplexity() for node in leaflist]
    #    nodeindex=0
    #    nodelist = self.getValidNodes()
    #    for node in leaflist:
    #        leafcount[nodeindex]=len(node.subdata)
    #        nodeindex+=1
    #    numberOfFeaturesUsed=self.getNumberOfFeaturesUsed()
    #    numberOfFeatureParameter=0.25
    #    loglikelihoodsum = -logfactorialdiff(0,numberOfFeaturesUsed)-logfactorialdiff(0,dimensionsize-numberOfFeaturesUsed)+numberOfFeaturesUsed*math.log(numberOfFeatureParameter)+(dimensionsize-numberOfFeaturesUsed)*math.log(1-numberOfFeatureParameter)-priorValue*len(nodelist)+math.log(priorValue)*(len(nodelist)-1)-sum([logfactorialdiff(0, node.getNumberOfBranch()-1)+logDirichletNormalizingDiff(2,[ len(self.tree[subnode].subdata) for subnode in node.getChild()]) for node in nodelist if node.getNumberOfBranch()>0])-sum([leafcount[i]*math.log(notUsedComplexityVector[i]) for i in range(len(leaflist))])
    #    return -loglikelihoodsum
    
    def counttrivialleaf(self):
        trivialleaf=0
        for node in self.tree:
            if node.leaf and node.active and len(node.subdata)==0:
                trivialleaf+=1
        return trivialleaf
    
    def countleafoftree(self):
        countleaf=0
        for node in self.tree:
            if node.leaf==True and node.active==True:
                countleaf+=1
        return countleaf
       

    def findneighbor(self,probabilitycutoff):
        #probability cutoff should be a vector of 2 values, say 0.99/2 and 0.99
        #anything less than that the first value, we do subtraction, anything between the two value, we do addition,
        #anything beyond that we do drastic change.
        #I need to handle special case of what if there is no parent to leaf node. The trivial model, in that case, just add node.
        randommove=random.random()
        candidatelist=[]
        candidateleaves=[]
        besidesroot=[node.me for node in self.tree if node.active and node.me>0]
        if randommove<probabilitycutoff[3]:
            for node in self.tree:
                if len(node.descendant)==len(node.child) and node.active==True and len(node.child)>0:
                    candidatelist.append(node) #note here that candidate list can be empty, candidate list is a list of nodes, wait, this is just for deletion, I shouldn't use this for addition
                elif len(node.descendant)== 0 and node.active==True and node.leaf==True:
                    candidateleaves.append(node)
            if randommove<probabilitycutoff[0] and len(candidatelist)>0:
                #print "move 1, remove node"
                self.removenode(random.choice(candidatelist).me)
            elif len(candidateleaves)<maxm and len(candidateleaves)>0 and randommove<probabilitycutoff[1] and randommove > probabilitycutoff[0]: #something to think about here..., as long as it is less than that number before addition
                randomcandidate=random.choice(candidateleaves)
                feature = random.choice(xrange(D))
                #print "move 2, adding node", randomcandidate.me
                self.addnode(randomcandidate.me, feature)
            elif len(besidesroot) > 1 and randommove< probabilitycutoff[2] and len(candidateleaves)<maxm-1 and randommove > probabilitycutoff[1]:
                nodeToBeSplitOn= random.choice(besidesroot)
                self.splitnode(nodeToBeSplitOn)
            elif len(candidatelist)>0 and randommove > probabilitycutoff[2]:
                parentNode = random.choice(candidatelist)
                if len(parentNode.child)>=3:
                    if parentNode.feature == 'cat':
                        node1, node2 = random.sample(parentNode.child, 2)
                        self.mergeNode(node1, node2)
                    elif len(parentNode.child)>0: #if parentNode.feature =='con':
                        #i need to perform a sorting
                        try:
                            drawTwo=[self.tree[kidChild].ancestorfeaturewithbinary[-1][1][0] for kidChild in parentNode.child]
                            #print "drawtwo",drawTwo
                        except:
                            print "drawtwo debugging", [self.tree[kidChild].ancestorfeaturewithbinary for kidChild in parentNode.child]
                        sortingIndex = [i[0] for i in sorted(enumerate(drawTwo), key=lambda x:x[1])]
                        tentativeIndex=random.choice(range(len(drawTwo)-1))
                        firstIndex = sortingIndex[tentativeIndex]
                        secondIndex = sortingIndex[tentativeIndex+1]
                        self.mergeNode(parentNode.child[firstIndex],parentNode.child[secondIndex])
                        #print "merging nodes"
        else:
            for node in self.tree:
                if len(node.descendant)>=2:
                    candidatelist.append(node)
            if len(candidatelist)>0:
                self.removenode(random.choice(candidatelist).me)
                #print "move 5 structural change"    

def simulatedannealingcrossvalidation(localdata,localtestdata,localvarietyMatrix,localD,localfeatureTypeVector,localvarietyDetailMatrix,localftrainVector,localftestVector,priorvector,toDoHellinger, splitindex,warmstartindicator):
    global data, testdata,varietyMatrix,D, featureTypeVector, varietyDetailMatrix, ndata, ftrainVector,ftestVector, branchobjective2, branchobjective3, objectivevector, leafcountvector, nodecountvector
    data = localdata
    testdata=localtestdata
    varietyMatrix = localvarietyMatrix
    D = localD
    featureTypeVector = localfeatureTypeVector
    varietyDetailMatrix = localvarietyDetailMatrix
    ndata = len(data)
    ftrainVector = localftrainVector
    ftestVector = localftestVector
    #global testdata
    objectivevector=[]
    branchobjective2=[]
    branchobjective3=[]
    leafcountvector = []
    nodecountvector = []
    crossvalidationvector=[simulatedannealing(data, priorparameter,warmstartindicator) for priorparameter in priorvector]
    #objectivefig = plt.figure()
    #plt.plot(objectivevector)
    #df=pd.DataFrame(objectivevector)
    #df.to_csv('objectivevector.csv',index=False,header=False)
    #plt.xlabel("index")
    #plt.ylabel("leaf based objective")
    #plt.savefig('objectiveevolution12'+str(splitindex)+'.jpg')

    crossvalidationObjectiveVector=[crossresult[1] for crossresult in crossvalidationvector]
    besttreesofar=crossvalidationvector[crossvalidationObjectiveVector.index(min(crossvalidationObjectiveVector))][0] #yup, extract the best tree
    #print "index chosen for leaf that is smaller is ", crossvalidationObjectiveVector.index(min(crossvalidationObjectiveVector))
    print "best leaf-based tree is"
    print besttreesofar.datadistribution(data)
    return [besttreesofar,besttreesofar.computeLikelihood(testdata)]

    #if toDoHellinger:
    #    return [besttreesofar.computeHellinger(testdata), besttreesofar.computeLikelihood(testdata), besttreesofar.countleafoftree(), besttreesofar.counttrivialleaf(), besttreesofar.computeHellingerTrain(data),besttreesofar.computeLikelihood(data), besttreesofar.crossvalidationForTest(testdata)]#min(crossvalidationObjectiveVector)]
    #else:
    #    return [-3.14, besttreesofar.computeLikelihood(testdata), besttreesofar.countleafoftree(), besttreesofar.counttrivialleaf(), -3.14,besttreesofar.computeLikelihood(data), besttreesofar.crossvalidationForTest(testdata)]#min(crossvalidationObjectiveVector)]

def simulatedannealing(data, priorparameter,warmstartindicator):
    global flagcount, testdata, branchobjective2, branchobjective3, objectivevector, leafcountvector, nodecountvector
    ndata=len(data)
    D=len(data[0])
    #mTree=2

    #create a tree of size m and with D possible features
    stoppingVectorLength = 50
    stoppingvector=[0]*stoppingVectorLength
    #start=time.time()

    mytree=Tree(D, data)
    if warmstartindicator:
        DETfile='DET_tree.txt'
        num_lines = sum(1 for line in open(DETfile))
        if num_lines>2:
            mytree.warmStart(D, data, DETfile) #this is to perform warm start
    #print mytree.datadistribution(data)
    likelihoodthreshold = mytree.computeLikelihood(data)
    #print likelihoodthreshold, "likelihoodthreshold"
    #df=pd.DataFrame(mytree.getDensityForTest(data))
    #df.to_csv('leafdensity.csv',index=False,header=False)
    #print "terminate program now"
    #################################################
    # Simulated Annealing
    ##################################################
    # Number of cycles
    n = 10#500
    # Number of trials per cycle
    m = 50#250#50
    # Number of accepted solutions
    na = 0.0
    # Probability of accepting worse solution at the start
    p1 = 0.7
    # Probability of accepting worse solution at the end
    p50 = 0.001
    # Initial temperature
    t1 = -1.0/math.log(p1)
    #t1=50000
    # Final temperature
    t50 = -1.0/math.log(p50)
    #t50=50000
    # Fractional reduction every cycle
    frac = (t50/t1)**(1.0/(n-1.0))
    # Initialize x
    mytreei=copy.deepcopy(mytree)
    na = na + 1.0
    # Current best results so far
    mytreec=copy.deepcopy(mytree)
    fc = mytreec.objectivevalue(data,priorparameter)
    # Current temperature
    t = t1
    # DeltaE Average
    DeltaE_avg = 0.0
    #leafvector=[]
    rejectCountVector=[]
    for i in range(n):
        for j in range(m):
            # Generate new trial points
            if (i%50 == 1 and i>100 and j==0):
                if random.random()<0.5:
                    mytreec=copy.deepcopy(besttreesofar)
                else:
                    mytreec=Tree(D, data)
            mytreei=copy.deepcopy(mytreec)
            mytreei.findneighbor(probabilityvector)
            mytreeiobjectivevalue=mytreei.objectivevalue(data, priorparameter)
            #if priorparameter == 4:
            #    objectivevector.append(mytreeiobjectivevalue)
            #    branchobjective2.append(mytreei.branchobjectivevalue(data, 2))
            #    branchobjective3.append(mytreei.branchobjectivevalue(data, 3))
            #    leafcountvector.append(len(mytreei.getValidLeaves()))
            #    nodecountvector.append(len(mytreei.getValidNodes()))
            DeltaE = abs(mytreeiobjectivevalue-fc)
            if (mytreeiobjectivevalue>fc):
                # Initialize DeltaE_avg if a worse solution was found
                #   on the first iteration
                if (i==0 and j==0): DeltaE_avg = DeltaE+0.000000001
                # objective function is worse
                # generate probability of acceptance
                if DeltaE_avg==0:
                    DeltaE_avg=10**(-16)
                p = math.exp(-DeltaE/(DeltaE_avg * t))   #i get division by zero sometimes, hmmm.....
                # determine whether to accept worse point
                if (random.random()<p):
                    # accept the worse solution
                    accept = True
                else:
                    # don't accept the worse solution
                    accept = False
                    #rejectCount+=1
            else:
                # objective function is lower, automatically accept
                accept = True
            #print "the active parts are", [activenode.me for activenode in mytreei.tree if activenode.active], "the accept status was", accept
            if (accept==True): # and mytreec.computeLikelihood(data) > 0.9*likelihoodthreshold:
                # update currently accepted solution
                mytreec=copy.deepcopy(mytreei)
                #print mytreec.countleafoftree()
                fc = mytreec.objectivevalue(data,priorparameter)
                #print fc
                # increment number of accepted solutions
                na = na + 1.0
                # update DeltaE_avg
                DeltaE_avg = (DeltaE_avg * (na-1.0) +  DeltaE) / na  
        #rejectCountVector.append(rejectCount/float(m))

        # Lower the temperature for next cycle
        t = frac * t
        #print mytreec
        if i==0:
            besttreesofar=copy.deepcopy(mytreec)
        elif mytreec.objectivevalue(data,priorparameter)<besttreesofar.objectivevalue(data,priorparameter): # and mytreec.computeLikelihood(data) > 0.9*likelihoodthreshold:
            besttreesofar=copy.deepcopy(mytreec)
        if i<stoppingVectorLength:
            stoppingvector[i]=mytreec.objectivevalue(data,priorparameter)
        else:
            for stoppingindex in range(stoppingVectorLength-1):
                stoppingvector[stoppingindex]=stoppingvector[stoppingindex+1]
            stoppingvector[stoppingVectorLength-1]=mytreec.objectivevalue(data,priorparameter)
            if sum([math.fabs(stoppingvector[terminationindex]-stoppingvector[terminationindex+1]) for terminationindex in range(stoppingVectorLength-1)])<10**(-31) and flagcount>10:
                break
    #print "my tree objective function", mytree.objectivevalue(data, priorparameter)
    #print "my final tree objective function", besttreesofar.objectivevalue(data,priorparameter)
    return [besttreesofar, besttreesofar.crossvalidationObjective(data)]

def load_data(fname):
    with open(fname+'.tab','r') as fin:
        A = fin.readlines()
    data = []
    for ln in A:
        data.append(ln.split())
    return data




