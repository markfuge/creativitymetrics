"""
variety_model.py

Defines the modular/submodular variety model used in the paper:
Mark Fuge, Josh Stroud, Alice Agogino. "Automatically Inferring Metrics for Design Creativity," in Proceedings of ASME 2013 International Design Engineering Technical Conferences & Computers and Information in Engineering Conference, August 4-2, 2013, Portland, USA
http://www.markfuge.com/papers/Fuge_DETC2013-12620.pdf

Authors: Josh Stroud and Mark Fuge
"""    
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

from concept_transforms import *

### Ported from mIDGenes/user_functions_public.js               ###

# Shah's variety metric: variety = summation(fj * summation(Sk*bk/n))
# fj :=  arbitrary weight based on judged importance of function fj
# Sk := weight for level k
# bk := # of nodes for level k
# n  := total number of concepts
# this metric is in concept space

# Assume here that fj is 1, and only 1 fj,
# eqn simplifies to: summation(Sk*bk/n)

def variety_shah_simplified(conceptTree):
    ''' Implements Shah et al.'s conceptual variety metric:
        "Metrics for measuring ideation effectiveness" 
        doi:10.1016/S0142-694X(02)00034-0
        
        Takes a concept tree as input and returns a numerical variety score
    '''
    assert conceptTree.spaceType == "concept", "tree argument not in concept space, please convert"

    # level weights from highest to lowest levels of abstraction
    if(conceptTree.numLevels == 4):
        level_weights = [10., 6., 3., 1.]
    else:
        level_weights = np.linspace(10.0, 1.0, conceptTree.numLevels)
    variety = 0.0

    for iter_level in xrange(0,len(level_weights)):
        Sk = level_weights[iter_level]
        bk = np.count_nonzero(conceptTree.set[iter_level])
        variety += Sk*bk
    variety /= conceptTree.numConcepts
    return variety

# Verhaegen's Pave Herfindahl variety metric: Vk = 10 * (1 / (n * summation(pi)))
# Idea-space metric (# of nodes versus # of trees)
# This metric calculates variety on a per-level basis (level number should be an argument)
# n := total number of ideas
# bk := # of nodes for level k
# pi := probability of each node i in summation from 1 to bk (for each node in level, basically)
# Herfindahl index is calculated for each level, it is the sum of the square of pi for each node

def variety_verhaegen(conceptTree, level):
    ''' Implements Verhaegen et al.'s conceptual variety metric:
        https://github.com/paularmand/mIDGenes
        
        Takes a concept tree as input and returns a numerical variety score
    '''
    assert conceptTree.spaceType == "idea", "tree argument not in idea space, please convert"
    herfindahl = 0
    for node in conceptTree.set[level]:
        probability_of_node = node / conceptTree.numConcepts
        herfindahl += probability_of_node * probability_of_node
    
    return 10 / (conceptTree.numConcepts * herfindahl)
    
def variety_verhaegen_tree(conceptTree,tree_depth=4):
    va = 0.0
    for i in xrange(0,tree_depth):
        va += variety_verhaegen(conceptTree,i)
    # Averages over tree depth to provide a normalized score
    return va/tree_depth

### End Ported Functions                ###
        
def noisyCompareTrees(A,B,percentError,metric):
    ''' Compares two concept trees, A and B, and returns 1, -1, or 0 depending 
        on which concept is greater.
        PercentError controls the amount of additional random noise which is added to both concepts.
        metric selects which variety function should be used to simulate users        
    '''
    if metric == 'verhaegen':
        vfunc = variety_verhaegen_tree
    else:
        vfunc = variety_shah_simplified
    
    # Get the ground truth variety
    va = vfunc(A)
    vb = vfunc(B)
    
    # Calculate the ground truth label for this comparison
    if(va > vb):
        ground_truth = 1
    elif(va < vb):
        ground_truth = -1
    else: 
        ground_truth = 0
    
    # Add error to the measurements to simulate noise
    nva = va + np.random.randn() * percentError
    nvb = vb + np.random.randn() * percentError
    
    # Produce the noisy training data
    if(nva > nvb):
        return (1,ground_truth)
    elif(nva < nvb):
        return (-1,ground_truth)
    else: 
        return (0,ground_truth)

def calcTreeRho(conceptTree, coverType = "set"):
    ''' Applying the submodular transform across each level of the tree consists
        of summing the submodular function across each sub branches of the tree.    
    '''
    l = len(conceptTree.set)
    returnVector = np.array([])
    for i in xrange(0,l):
        returnVector = np.hstack([returnVector, rho_func(conceptTree.set[i], coverType)])
    return returnVector

def rho_func(vector, coverType = "set") :
    ''' Transforms a given vector using a submodular set transformation, given
        by coverType.    
    '''
    sum = 0.
    # theta represents a hyperparameter that could be optimized in the future
    theta = .75
    i = 0
    for v in vector:
        rv = 0.
        if(coverType == "set"):
            if(v > 0):
                rv = 1.
        elif(coverType == "prob"):
            theta = .75
            rv = 1. - np.exp(-theta*v)
        elif(coverType == "log"):
            theta = 1.5
            rv = np.log(theta*v + 1)
        elif(coverType == "mod"):
            if(v > 0):
                rv = v
        else:
            raise SyntaxError("wrong cover type in rho_func")
        sum += rv
        i += 1
    return sum 
    
def generate_random_trees(numLevels = 4,numConceptsPerTree = 10, numTrees = 1000):   
    ''' Randomly constructs sets of concept trees for use during testing'''
    treeArray = []
    for i in xrange(0,numTrees):
        treeArray.append(genRandConceptTree(numConceptsPerTree))
    return treeArray
    
def remove_zero_training_labels(X,Y,Ytrue):
    ''' Since the logistic regression is just a binary classifier, we throw out 
        any comparisons where the variety is equal. This reduces the amount of 
        training data somewhat, but not by much. Future work involves improving the
        model to handle the equal variety case.'''
    ind = np.where(Y[:,0]==0)[0]
    Y=np.delete(Y,ind,axis=0)
    Ytrue=np.delete(Ytrue,ind,axis=0)
    X=np.delete(X,ind,axis=0)            
    return X,Y,Ytrue

def generate_comparison_data(numConceptsPerTree = 10, numComparisons = 1000, 
                             numLevels = 4, cover_type = None, metric = "shah", E = [0.0]):
    ''' Generates two data matrices for training the model.
        X is a numComparisons by numFeatures matrix
        Y is a numComparisons by len(E) matrix
           - Each coefficient in E corresponds to one column of Y. This way you
             can use different columns of Y to test different error amounts
    '''
    if(metric == "shah"):
        #compareFunc = compareTreesShah
        #noisyCompareFunc = noisyCompareTreesShah
        spaceType = "concept"
        if(not cover_type):
            cover_type = "set"
    elif(metric == "verhaegen"):
        #compareFunc = compareTreesVerhaegen
        #noisyCompareFunc = noisyCompareTreesVerhaegen
        spaceType = "ideas"
        if(not cover_type):
            cover_type = "prob"
    
    # Time to build the data arrays
    numRho = numLevels
    num_error_coeffs = len(E)
    # Preallocating Array space
    X = np.zeros([numComparisons, numRho])
    Y = np.zeros([numComparisons,num_error_coeffs])
    Ytrue = np.zeros([numComparisons,num_error_coeffs])
    # Generate the data
    for i in xrange(0,numComparisons):
        # Go through each pair and determine the feature vector and the ratings
        A = genRandConceptTree(numConceptsPerTree, numLevels = numLevels)
        B = genRandConceptTree(numConceptsPerTree, numLevels = numLevels)
        if spaceType == "concept":
            A.convertToConceptSpace()
            B.convertToConceptSpace()
        # Calculate submodular feature vector:
        X[i] = calcTreeRho(A, cover_type) - calcTreeRho(B, cover_type)
        # Calculate training votes
        for j in xrange(0,num_error_coeffs):
            Y[i][j],Ytrue[i][j] = noisyCompareTrees(A,B,E[j],metric)
        
    # Normalizes the matrix - not strictly necessary, but improves performance
    X = X/ float(numConceptsPerTree)
    
    # Remove any 0 training cases to binarize the data
    X,Y,Ytrue = remove_zero_training_labels(X,Y,Ytrue)
    
    return X,Y,Ytrue
    
def runSklearn(X,Y,Ytrue,numTrain,numRetest = 5):
    ''' Conducts actual model inference using Scikit-learn's logistic regression
    '''
    interceptOn = False

    # Split the dataset up into segments that have numTrain # of training
    # examples. Rerun the experiment numRetest number of times, for robustness
    # We set test_size to a constant only for consistency across experiment cases
    n = len(X)
    datasamples = cross_validation.ShuffleSplit(n,
                                                n_iter=numRetest,
                                                test_size=7500,
                                                train_size=numTrain,
                                                random_state=0)    
    
    tolerance = 0.0001  # Specifies final covergence of coefficients
    C = 1.    # Specifies the strength of the regularization,
              # the smaller it is the bigger is the regularization.
              # Currently this is not optimized, though it could be in the future.

    intercept = 0.0001
    # if interceptOn is true, enable default intercept = 1.0 
    clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=tolerance, fit_intercept=interceptOn, intercept_scaling = intercept)
    # If you would prefer to try L2 regularized logistic regression, you can 
    # uncomment the below line. For instances where you are unsure of the weight
    # distribution, L2 would be the correct choice. Since we are expecting different
    # weight distributions across the features, we use L1 regularization.
    #clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=tolerance, fit_intercept=interceptOn, intercept_scaling = intercept)
    
    # Run a cross-validated estimator over the L1 Logistic Regression
    numErr = len(Y[0])
    errScores = []
    gterrScores = []
    for i in xrange(0,len(Y[0])):
        cv_scores = []
        gtcv_scores = []
        y = Y[:,i]
        ytrue = Ytrue[:,i]
        for train_index, test_index in datasamples:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            ytrue_test = ytrue[test_index]
            clf_l1_LR.fit(X_train,y_train)
            cv_scores.append(clf_l1_LR.score(X_test,y_test))
            gtcv_scores.append(clf_l1_LR.score(X_test,ytrue_test))
        cv_scores = np.array(cv_scores)
        gtcv_scores = np.array(gtcv_scores)
        errScores.append((cv_scores.mean(), cv_scores.std()))
        gterrScores.append((gtcv_scores.mean(), gtcv_scores.std()))
    return errScores, gterrScores
    
### Older example and test code which might be helpful in understanding some of
### The fundamental functions
    
def examples():
    '''
        Demonstrate how to use the ConceptTree class and the variety model to
        predict how random trees will be used for training and prediction.
    '''
    numConcepts = 5
    
    print "* start test"
    print "** testing idea space generation..."
    for i in xrange(0,5): 
        print "*** generating",numConcepts,"concepts into a tree..."
        concepts = genRandConceptTree(numConcepts)
        shah_score = variety_shah_simplified(concepts)
        print "random concept set:\n", concepts.printSet()
        print "shah's variety score on set: ", shah_score

    numComparisons = 10 
    print "** testing tree pair comparison..."
    print "*** generating and comparing", numComparisons, "trees..."
    
    compareVector = np.zeros([numComparisons])
    for i in xrange(0,numComparisons):
        (A,B) = genTwoTrees()    
        compareVector[i] = compareTreesShah(A,B)
     
    print "results vector: ", compareVector
    
    print "** testing rho function..."
    
    print "*** set cover"
	# should compare tree, not compareVector
    rho = rho_func(compareVector, "set")
    print rho
    
    print "*** probabilistic cover"
    rho = rho_func(compareVector, "prob")
    print rho

    print "*** logarithmic cover"
    rho = rho_func(compareVector, "log")
    print rho   
    
def noisyCompareTreesShah(A, B, percentError):
    va = variety_shah_simplified(A)
    va += np.random.randn() * percentError

    vb = variety_shah_simplified(B)
    vb += np.random.randn() * percentError
    
    if(va > vb):
        return 1
    elif(va < vb):
        return -1
    else: 
        return 0

def compareTreesShah(A, B): 

    va = variety_shah_simplified(A)
    vb = variety_shah_simplified(B)
    
    if(va > vb):
        return 1
    elif(va < vb):
        return -1
    else: 
        return 0
        
def compareTreesVerhaegen(A, B):
    va = 0.0
    vb = 0.0
    for i in xrange(0,4):
        va += variety_verhaegen(A,i)
        vb += variety_verhaegen(B,i)

    if(va > vb):
        return 1
    elif(va < vb):
        return -1
    else: 
        return 0
        
def noisyCompareTreesVerhaegen(A, B, percentError):
    va = 0.0
    vb = 0.0
    for i in xrange(0,4):
        va += variety_verhaegen(A,i)
        vb += variety_verhaegen(B,i)

    va += np.random.randn() * percentError
    vb += np.random.randn() * percentError
    
    if(va > vb):
        return 1
    elif(va < vb):
        return -1
    else: 
        return 0