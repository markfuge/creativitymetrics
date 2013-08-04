"""
paper_experiments.py

Uses other modules to re-produce the results and paper graphs contained 
in the paper. Authors wanting to reproduce or compare to our algorithm can
run the experiments by executing:
python paper_experiments.py
from the command line

Paper:
Mark Fuge, Josh Stroud, Alice Agogino. "Automatically Inferring Metrics for Design Creativity," in Proceedings of ASME 2013 International Design Engineering Technical Conferences & Computers and Information in Engineering Conference, August 4-2, 2013, Portland, USA
http://www.markfuge.com/papers/Fuge_DETC2013-12620.pdf

Authors: Josh Stroud and Mark Fuge
"""
import os
import numpy as np
import pylab as pl

from variety_model import *

# Where to save plots
plot_path = "plots/"

# For output plots
def setfont():
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size' : 20}
    pl.matplotlib.rc('font', **font) 

def genConvergencePlots(metric="shah", numLevels = 4, cover_type = None):
    ''' Generates various sensitivity plots presented in the paper.
        Notably,'''
    x = []
    y = []
    xplots = []
    yplots = []

    ### SCRIPT PARAMETERS 
    errorCoeffs = [0, 1, 2, 3, 5]
    plotFlag = 1
    
    numRepeat = 3   # Amount of resampling - increasing this will improve the
                    # statistical reliability of the resulting accuracy estimates,
                    # at the cost of additional computation time.
    numSamples = 50 # Sets the x-scale fidelity of the convergence plots (Figs. 3-5)
                    # Increasing this will increase the number of experiments conducted
                    # thus increasing run time
    
    # Init some array storage
    xmat = np.zeros([numSamples])
    ymat = np.zeros([numSamples,len(errorCoeffs)])
    yerr = np.zeros([numSamples,len(errorCoeffs)])
    ytmat = np.zeros([numSamples,len(errorCoeffs)])
    yterr = np.zeros([numSamples,len(errorCoeffs)])
    
    # close any open plot windows
    pl.close('all')
    
    if not cover_type:
        if metric == 'shah':
            cover_type = 'set'
        elif metric == 'verhaegen':
            cover_type = 'prob'
    
    print "generating convergence plots"
    print "using",metric,"variety metric, numLevels =",numLevels
    if(cover_type):
        print "Cover Type:",cover_type
    else:
        print "Cover Type: Default"
        
    # One-time generation of all the random tree samples.
    # We'll then partion up the dataset such that we use only the required
    # fraction of training samples for the model.
    max_comparisons = 10000
    numConceptsPerTree = 10
    
    # Generates the data
    print "Generating Data Samples..."
    X,Y,Ytrue = generate_comparison_data(numConceptsPerTree = numConceptsPerTree,
                                numComparisons = max_comparisons,
                                metric = metric,
                                cover_type = cover_type,
                                E = errorCoeffs,
                                numLevels = numLevels)
    
    # Now we have generated all of the simulated concept sets, as well as
    # All of the noisy ratings and true ratings. We can now run the experiments
    print "Running Experiments..."
    
    # This will determine the range of comparisons we will test over.
    xmat = np.round(np.linspace(0 , 1500, numSamples+1))
    xmat = xmat[1:]
    # Runs the model Training and Evaluation
    for j, numTraining in enumerate(xmat):
        numTraining = int(numTraining)
        if(j % 10 == 0):
            print "Processing sample",j,"/",numSamples
        # Run the model
        errScores,gterrScores = runSklearn(X,Y,Ytrue, numTraining,numRetest=numRepeat)
        # errScores now contains an array of tuples (mean, std) of the scores across
        # numRetest runs of the data
        
        if(plotFlag == 1):
            for i,e in enumerate(errorCoeffs):
                ymat[j,i] = errScores[i][0]
                yerr[j,i] = errScores[i][1]
                ytmat[j,i] = gterrScores[i][0]
                yterr[j,i] = gterrScores[i][1]

    # Print out a sample of accuracy point estimates
    print "Final accuracy for metric: "+metric
    for i in range(0,numSamples,numSamples/10)[1:]:
        print "n: %d\tacc: %.1f"%(xmat[i],100*ymat[i,0])
    
    # Now do the plotting
    if(plotFlag == 1):
        method_fig = pl.figure(metric+' Training Convergence')
        pl.hold(True)
        x = xmat
        for i,e in enumerate(errorCoeffs):
            #pl.plot(x,ymat[:,i],'-',label='E = ' + str(e))
            # uncomment for 95% confidence interval
            pl.errorbar(x,ymat[:,i],yerr[:,i]*1.96,label=r'$\sigma$'+': ' + str(e))
        pl.hold(False)
        pl.xlabel("Number of A/B Comparisons used in training")
        pl.ylabel("Noisy Label Prediction accuracy")
        pl.title(metric+" Training, levels:"+ str(numLevels)+", cover:"+cover_type)
        pl.ylim((.5,1.0))
        pl.xlim((0,x[-1]))
        pl.legend(loc=4,prop={'size':14})
        # uncomment below if you want interactive plotting
        #pl.show()
        pl.savefig(plot_path +
                   "metric=" + metric +
                   "_numLevels=" + str(numLevels) + 
                   "_cover=" + cover_type + 
                   "_training_convergence.pdf")
        
        method_fig = pl.figure(metric+' Ground Truth Convergence')
        pl.hold(True)
        for i,e in enumerate(errorCoeffs):
            pl.plot(x,ytmat[:,i],label=r'$\sigma$'+': ' + str(e))
            # uncomment for 95% confidence interval
            #pl.errorbar(x,ytmat[:,i],yterr[:,i]*1.96,label=r'$\sigma$'+': ' + str(e))
            
        pl.hold(False)
        pl.xlabel("Number of A/B Comparisons used in training")
        pl.ylabel("Ground Truth Prediction accuracy")
        pl.title(metric+" Truth, levels:"+ str(numLevels)+", cover:"+cover_type)
        pl.ylim((.5,1.0))
        pl.xlim((0,x[-1]))
        pl.legend(loc=4,prop={'size':14})
        # uncomment below if you want interactive plotting
        #pl.show()
        pl.savefig(plot_path +
                   "metric=" + metric +
                   "_numLevels=" + str(numLevels) + 
                   "_cover=" + cover_type + 
                   "_groundtruth_convergence.pdf")
           
    print "Completed Convergence Experiment!\n"
    return xmat,ymat
    
def genExperimentalResults():
    ''' Generates the main experimental results and figures used in the paper
    '''

    shahx,shahy = genConvergencePlots("shah",cover_type="set")
    shah_prob_x,shah_prob_y = genConvergencePlots("shah",cover_type="prob")
    verhx,verhy = genConvergencePlots("verhaegen",cover_type="prob")
    verh_set_x,verh_set_y = genConvergencePlots("verhaegen",cover_type="set")
    
    x = shahx
    compare_fig = pl.figure('Convergence of different metrics')
    pl.hold(True)
    pshah = pl.plot(shahx,shahy[:,0],'k-',label="Shah (Set)",linewidth=3)
    pshah_prob = pl.plot(shah_prob_x,shah_prob_y[:,0],'k-',label="Shah (Prob)",linewidth=1)
    pverh_set = pl.plot(verh_set_x,verh_set_y[:,0],'b--',label="Verhaegen (Set)",linewidth=3)
    pverh = pl.plot(verhx,verhy[:,0],'b--',label="Verhaegen (Prob)",linewidth=1)    
    pl.hold(False)
    pl.xlabel("Number of A/B Comparisons used in training")
    pl.ylabel("Prediction accuracy")
    pl.title("Comparison of various metrics")
    pl.ylim((.5,1.0))
    pl.xlim((0,shahx[-1]))
    pl.legend(loc=4)
    # Uncomment if you want interactive plotting
    #pl.show()
    pl.savefig(plot_path+"metric_convergence_comparison.pdf")
    
def genSensitivityResults():
    ''' Generates sensitivity results regarding number of tree levels and how
        increasing the number of estimation parameters affects convergence.
        Didn't have space to include these figures in the conference paper.
    '''
 
    shahx_a,shahy_a = genConvergencePlots("shah", numLevels=4)
    shahx_b,shahy_b = genConvergencePlots("shah", numLevels=10)
    shahx_c,shahy_c = genConvergencePlots("shah", numLevels=25)
    shahx_d,shahy_d = genConvergencePlots("shah", numLevels=50)
    
    compare_fig = pl.figure('Convergence of different metrics')
    pl.hold(True)
    shaha = pl.plot(shahx_a,shahy_a[:,0],'k-',label="# Shah Levels = 2",linewidth=3)
    shahb = pl.plot(shahx_b,shahy_b[:,0],'k-',label="# Shah Levels = 4",linewidth=1)
    shahc = pl.plot(shahx_c,shahy_c[:,0],'b--',label="# Shah Levels = 10",linewidth=3)
    shahd = pl.plot(shahx_d,shahy_d[:,0],'b--',label="# Shah Levels = 50",linewidth=1)    
    pl.hold(False)
    pl.xlabel("Number of A/B Comparisons used in training")
    pl.ylabel("Prediction accuracy")
    pl.title("Comparison of various metrics")
    pl.ylim((.5,1.0))
    pl.xlim((0,shahx_a[-1]))
    pl.legend(loc=4)
    # Uncomment if you want interactive plotting
    #pl.show()
    pl.savefig(plot_path+"sensitivity_convergence_comparison.pdf")

# script below to generate plots
if __name__ == "__main__":
    setfont()
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    genExperimentalResults()
    genSensitivityResults()
