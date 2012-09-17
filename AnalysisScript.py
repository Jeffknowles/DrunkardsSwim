import matplotlib.pylab as plt
import numpy as np
import scipy as sp
import AnalysisFunctions as af



## Choose flow conditions:

FLOW = 'Off'

## Choose what to analyze and plot:

ORIENT_ANALYSIS = 'Off'
KL_Analysis = 'On'
Descriptive_Statistics = 'On'
BayesHypothesisTest = 'On'


## First import files.
if FLOW == 'Off':
    
    DATA = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Combined_Data/JEBdataNoFlowCombined.csv', delimiter=',')
    MODEL1 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/DrunkardsSwim/New_Flow/Model_NEW10cm_LL_NoFlow.csv',
                           delimiter=',')
    MODEL2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/DrunkardsSwim/JEB_Flow/Model_JEB10cm_LLNoFlow.csv',
                           delimiter=',')

if FLOW == 'On':
    DATA = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Combined_Data/JEBFlowCombined.csv', delimiter=',')
    MODEL1 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/DrunkardsSwim/New_Flow/Model_NEW10cm_LL_Flow.csv',
                      delimiter=',')
    MODEL2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Combined_Data/JEBdataNoFlowCombined.csv', delimiter=',')


## Orientation.

if ORIENT_ANALYSIS == 'On':
    
    Data_Orient,bins = af.OrientationHist(DATA)
    Model1_Orient,bins = af.OrientationHist(MODEL1)
    Model2_Orient,bins = af.OrientationHist(MODEL2)

    af.OrientationPlot(Data_Orient,bins)
    af.OrientationPlot(Model1_Orient,bins)
    af.OrientationPlot(Model2_Orient,bins)

## Standardize XYZ Data.

Data_Standard = af.StandardizeData(DATA,68.0, 15.0, Loc=np.array([0,1,2]))
Model1_Standard = af.StandardizeData(MODEL1, 68.0, 15.0)
Model2_Standard = af.StandardizeData(MODEL2, 68.0, 15.0)

## Descriptive statistics: ProportionDown()

if Descriptive_Statistics == 'On':
    
    Data_Downstream = af.ProportionDown(Data_Standard)
    Model1_Downstream = af.ProportionDown(Model1_Standard)
    Model2_Downstream = af.ProportionDown(Model2_Standard)

    if FLOW == 'On':
        af.PropDownstreamPlot(Data_Downstream,Model1_Downstream,Model2_Downstream)
    else :
        af.PropDownstreamPlot(Data_Downstream,Model1_Downstream,Model2_Downstream,YLIMIT=[0.,0.75])

## KL-Divergence.

if KL_Analysis == 'On':

    # X Dimension
    print 'X dimension'
    Model1_KL = af.KL_Diverg(Data_Standard,Model1_Standard,Dim=0)
    Model2_KL = af.KL_Diverg(Data_Standard,Model2_Standard,Dim=0)

    # Y Dimension
    print 'Y dimension'
    Model1_KL = af.KL_Diverg(Data_Standard,Model1_Standard,Dim=1)
    Model2_KL = af.KL_Diverg(Data_Standard,Model2_Standard,Dim=1)

    # Z Dimension
    print 'Z dimension'
    Model1_KL = af.KL_Diverg(Data_Standard,Model1_Standard,Dim=2)
    Model2_KL = af.KL_Diverg(Data_Standard,Model2_Standard,Dim=2)
    
## Bayes Hypothesis Testing.

if BayesHypothesisTest == 'On':
    Data_3DProb = af.ProbabilityDist3D(Data_Standard)
    Model1_3DProb = af.ProbabilityDist3D(Model1_Standard)
    Model2_3DProb = af.ProbabilityDist3D(Model2_Standard)

# Not working
BF_01,BF_02,BF_12 = af.BayesHypothesisTest(Data_Standard,Model1_3DProb,Model2_3DProb)

## Plot data.


# output of MarginalProbabilityDist is not quite right, particularly for Data_Standard

X_Data,Y_Data,Z_Data = af.MarginalProbabilityDist(Data_Standard)
X_Model1,Y_Model1,Z_Model1 = af.MarginalProbabilityDist(Model1_Standard)
X_Model2,Y_Model2,Z_Model2 = af.MarginalProbabilityDist(Model2_Standard)

af.X_DimensionPlot(X_Data.T,X_Model1.T,X_Model2.T)
af.YZ_DimensionPlot(Y_Data.T,Z_Data.T,Y_Model1.T,Z_Model1.T,Y_Model2.T,Z_Model2.T)

