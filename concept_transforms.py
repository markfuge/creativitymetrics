"""
concept_transforms.py

Utility functions for transforming design concepts into variety feature vector

Paper:
Mark Fuge, Josh Stroud, Alice Agogino. "Automatically Inferring Metrics for Design Creativity," in Proceedings of ASME 2013 International Design Engineering Technical Conferences & Computers and Information in Engineering Conference, August 4-2, 2013, Portland, USA
http://www.markfuge.com/papers/Fuge_DETC2013-12620.pdf

Authors: Josh Stroud and Mark Fuge
"""
import random
import numpy as np

class ConceptTree: 
    ''' Concept Tree class
         The Concept Tree defines a Shah-like deconstruction of a set of concepts
         into a hierarchical tree. This tree hierarchies are historically
         functionally based, though it could be any hierarchical decomposition.
    '''
    def __init__(self, set=np.array([ [],[],[],[] ]), numConcepts=2, numLevels = 4, spaceType="idea", E=0.0):
        self.set = set
        self.numConcepts = np.sum(set[0])
        self.numLevels = numLevels
        self.numIdeas = self.calcNumIdeas()
        self.spaceType = spaceType
        self.E = E

    def printSet(self):
        print self.set

    def calcNumIdeas(self):
        num = 0
        for level in self.set:
            num += np.count_nonzero(level)
        return num

    def convertToIdeaSpace(self):
        if(self.spaceType == "concept"):
            for level in xrange(0, self.numLevels-1):
                i = 0
                for parent in self.set[level]:
                    while(parent > 0):
                        child = self.set[level+1][i] 
                        if(child == 0):
                            self.set[level+1][i] = 1
                            parent -= 1
                        else:
                            parent -= child
                        i += 1
            self.spaceType = "idea"

    def convertToConceptSpace(self):
        if(self.spaceType == "idea"):
            newSet = self.set.copy()
            for level in xrange(0, self.numLevels-1):
                i = 0
                for parent in self.set[level]:
                    only = True
                    while(parent > 0 and i < self.numConcepts):
                        child = self.set[level+1][i] 
                        if(only == True and child == 1 and parent == 1):
                            newSet[level+1][i] = 0
                            parent = 0
                        else:
                            only = False
                            parent -= child
                        i += 1
            self.set = newSet
            self.spaceType = "concept"
        
        
# generates random concept trees in the concept space
def genRandConceptTree(numConcepts = 10, ideaVariety = 5, numLevels = 4):
    set = np.zeros([numLevels, numConcepts])
    i = 0

    numChildren = numConcepts
    while(i < numLevels):
        if(i == 0):
            k = 0
            while(numChildren > 0):
                ind = random.randint(1, numChildren)
                set[i,k] = ind
                numChildren -= ind
                k += 1
        else:
            k = 0
            for numChildren in set[i-1]:
                while(numChildren > 0):
                    ind = random.randint(1, numChildren)
                    set[i,k] = ind
                    numChildren -= ind
                    k += 1
        i += 1
    return ConceptTree(set, numConcepts, spaceType = "idea", numLevels = numLevels)

