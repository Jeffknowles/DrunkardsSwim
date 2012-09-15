import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

## Still in development ##
## Need to add 3d distributions. ##


DATA1 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Combined_Data/NewNoFlowCombined.csv', delimiter=',')
MODEL = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/New_Flow/Model_NEW10cm_LL_NoFlow.csv',
                      delimiter=',')
DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Combined_Data/AllExpCombinedNoFlow.csv', delimiter=',')
DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Combined_Data/JEBdataNoFlowCombined.csv', delimiter=',')


Real_New = (DATA1[:,4],DATA1[:,5],DATA1[:,6])
P_New = np.zeros(len(Real_New[0])/10)
for i in range(0,len(Real_New[0])/10):
    P_New[i] = np.var(Real_New[0][10*i:(10*i)+10])


Real_JEB = (DATA3[:,0],DATA3[:,1],DATA3[:,2])
P_JEB = np.zeros(len(Real_JEB[0])/10)
for i in range(0,len(Real_JEB[0])/10):
    P_JEB[i] = np.var(Real_JEB[0][10*i:(10*i)+10])

Real_Comb = (DATA2[:,0],DATA2[:,1],DATA2[:,2])
P_Comb = np.zeros(len(Real_Comb[0])/10)
for i in range(0,len(Real_Comb[0])/10):
    P_Comb[i] = np.var(Real_Comb[2][10*i:(10*i)+10])


Model = (MODEL[:,0],MODEL[:,1],MODEL[:,2])
P_Model = np.zeros(len(Model[0])/10)
for i in range(0,len(Model[0])/10):
    P_Model[i] = np.var(Model[0][10*i:(10*i)+10])


## Compute Probabilities

BINS = np.linspace(min(min(P_New),min(P_Model),min(P_JEB),min(P_Model)),
                   max(max(P_New),max(P_Model),max(P_JEB),max(P_Model)),75)

# find distributions
Pr_New,q = np.histogram(P_New,bins=BINS)
Pr_JEB,q = np.histogram(P_JEB,bins=BINS)
Pr_Comb,q = np.histogram(P_Comb,bins=BINS)
Pr_Model,q = np.histogram(P_Model,bins=BINS)

# prevent zeros in division 
Pr_New += 1
Pr_JEB += 1
Pr_Comb += 1
Pr_Model += 1

# find probabilities
Prob_New = np.true_divide(Pr_New,sum(Pr_New))
Prob_JEB = np.true_divide(Pr_JEB,sum(Pr_JEB))
Prob_Comb = np.true_divide(Pr_Comb,sum(Pr_Comb))
Prob_Model = np.true_divide(Pr_Model,sum(Pr_Model))

# compute KL-Divergence between Model and Observed Distributions.
KLdiverg_NewModel = 0
KLdiverg_JEBModel = 0
KLdiverg_CombModel = 0
for i in range(0,len(BINS)-1):
    KLdiverg_NewModel += Prob_New[i]*np.log2(Prob_New[i]/Prob_Model[i])
    KLdiverg_JEBModel += Prob_JEB[i]*np.log2(Prob_JEB[i]/Prob_Model[i])
    KLdiverg_CombModel+= Prob_Comb[i]*np.log2(Prob_Comb[i]/Prob_Model[i])


print 'New Data       |       JEB Data       |      Combined data '
print '______________________________________________________________'
print KLdiverg_NewModel, '    ', KLdiverg_JEBModel, '      ', KLdiverg_CombModel
