import matplotlib.pylab as plt
import numpy as np
import scipy as sp
import AnalysisFunctions as af

## This script will be used to call the Analysis Functions.

## First import files.

FLOW = 'Off'
ORIENT_ANALYSIS = 'Off'
KL_Analysis = 'Off'
Descriptive_Statistics = 'Off'


if FLOW == 'Off':
    
    DATA = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Combined_Data/NewNoFlowCombined.csv', delimiter=',')
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
    Data_Orient,bins = af.OrientationHist(DATA,Loc=7)
    Model1_Orient,bins = af.OrientationHist(MODEL1)
    Model2_Orient,bins = af.OrientationHist(MODEL2)

    af.OrientationPlot(Data_Orient,bins)
    af.OrientationPlot(Model1_Orient,bins)
    af.OrientationPlot(Model2_Orient,bins)

## Standardize XYZ Data.

Data_Standard = af.StandardizeData(DATA,68.0, 15.0, Loc=np.array([4,5,6]))
Model1_Standard = af.StandardizeData(MODEL1, 68.0, 15.0)
Model2_Standard = af.StandardizeData(MODEL2, 68.0, 15.0)

## Descriptive statistics: ProportionDown()

if Descriptive_Statistics == 'On':
    Nothing = 1

## KL-Divergence.

if KL_Analysis == 'On':
    A=1
    
## Bayes Hypothesis Testing.

Data_3DProb = af.ProbabilityDist3D(Data_Standard)
Model1_3DProb = af.ProbabilityDist3D(Model1_Standard)
Model2_3DProb = af.ProbabilityDist3D(Model2_Standard)

BF_01,BF_02,BF_12 = af.BayesHypothesisTest(Data_Standard,Model1_3DProb,Model2_3DProb)

## Plot data.

# output of MarginalProbabilityDist is not quite right, particularly for Data_Standard

X_Data,Y_Data,Z_Data = af.MarginalProbabilityDist(Data_Standard)
X_Model1,Y_Model1,Z_Model1 = af.MarginalProbabilityDist(Model1_Standard)
X_Model2,Y_Model2,Z_Model2 = af.MarginalProbabilityDist(Model2_Standard)

af.X_DimensionPlot(X_Data.T,X_Model1.T,X_Model2.T)
af.YZ_DimensionPlot(Y_Data.T,Z_Data.T,Y_Model1.T,Z_Model1.T,Y_Model2.T,Z_Model2.T)
