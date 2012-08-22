import matplotlib.pyplot as plt
import pylab as plb
import numpy as np
import scipy as sp
import csv as csv
#import os as os
#from matplotlib.ticker import LogLocator

FLOW = 0



if FLOW == 0:
    DATA1 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Spreadsheets/JEB_paper/10cm/Experimental_Data/Pulsed_T1F1.csv', delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Spreadsheets/JEB_paper/10cm/Model_Tadpole/FrogModelNo_LL10cm.csv', delimiter=',')
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Spreadsheets/JEB_paper/10cm/Model_Tadpole/FrogModelLL10cm.csv', delimiter=',')
if FLOW == 1:
    DATA1 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Spreadsheets/JEB_paper/10cm/Experimental_Data/Pulsed_T1F2.csv',delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Spreadsheets/JEB_paper/10cm/Model_Tadpole/FrogModelFlowNo_LL10cm.csv',delimiter=',')    
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/Spreadsheets/JEB_paper/10cm/Model_Tadpole/FrogModelFlowLL10cm.csv', delimiter=',')

## Calculate Mean and Probability

###################
###Measured Data###
###################

Real = (DATA1[:,5]*2.54,DATA1[:,6]*2.54,DATA1[:,7]*2.54)

## np.histogram Bins
X_BINS = np.arange(0,70.1,(70.0/27.0))
Y_BINS = np.arange(0,16.1,(16/6.0))
Z_BINS = np.arange(0,16.1,(16/6.0))

## X DIMENSION
Real_hist_X = np.zeros((10,len(X_BINS)-1))
Real_X = np.zeros((10,2))
for i in range(0,10):
    Real_X[i,:] = [ np.mean(Real[0][i::10]),np.std(Real[0][i::10]) ]
    PX,bins,ignore = plt.hist(Real[0][i::10],bins=X_BINS); plt.close()
    Real_hist_X[i,:] = PX
Prob_Real_X = Real_hist_X/(len(Real[0])/10)

## Y DIMENSION
Real_hist_Y = np.zeros((10,len(Y_BINS)-1))
Real_Y = np.zeros((10,2))
for i in range(0,10):
    Real_Y[i,:] = [ np.mean(Real[1][i::10]),np.std(Real[1][i::10]) ]
    PY,bins,ignore = plt.hist(Real[1][i::10],bins=Y_BINS); plt.close()
    Real_hist_Y[i,:] = PY
Prob_Real_Y = Real_hist_Y/(len(Real[1])/10)

## Z DIMENSION
Real_hist_Z = np.zeros((10,len(Z_BINS)-1))
Real_Z = np.zeros((10,2))
for i in range(0,10):
    Real_Z[i,:] = [ np.mean(Real[2][i::10]),np.std(Real[2][i::10]) ]
    PZ,bins,ignore = plt.hist(Real[2][i::10],bins=Z_BINS); plt.close()
    Real_hist_Z[i,:] = PZ
Prob_Real_Z = Real_hist_Z/(len(Real[2])/10)


######################################
#### RANDOM WALK MODEL DATA - M1 #####
######################################
Model = (DATA2[:,0],DATA2[:,1],DATA2[:,2])

a = (Model[0]/max(Model[0]))*(68.5-2.54)+2.54
b = (Model[1]/max(Model[1]))*(15.24-2.54)+2.54
c = (Model[2]/max(Model[2]))*(15.24-2.54)+2.54

Model = (a,b,c)

## X DIMENSION
hist_X = np.zeros((10,len(X_BINS)-1))
Model_X = np.zeros((10,2))
for i in range(0,10):
    Model_X[i,:] = [ np.mean(Model[0][i::10]),np.std(Model[0][i::10]) ]
    P,bins,ignore = plt.hist(Model[0][i::10],bins=X_BINS); plt.close()
    hist_X[i,:] = P
Prob_Model_X = hist_X/(len(Model[0])/10)

## Y DIMENSION
hist_Y = np.zeros((10,len(Y_BINS)-1))
Model_Y = np.zeros((10,2))
for i in range(0,10):
    Model_Y[i,:] = [ np.mean(Model[1][i::10]),np.std(Model[1][i::10]) ]
    P,bins,ignore = plt.hist(Model[1][i::10],bins=Y_BINS); plt.close()
    hist_Y[i,:] = P
Prob_Model_Y = hist_Y/(len(Model[0])/10)
 
## Z DIMENSION
hist_Z = np.zeros((10,len(Z_BINS)-1))
Model_Z = np.zeros((10,2))
for i in range(0,10):
    Model_Z[i,:] = [ np.mean(Model[2][i::10]),np.std(Model[2][i::10]) ]
    P,bins,ignore = plt.hist(Model[2][i::10],bins=Z_BINS); plt.close()
    hist_Z[i,:] = P
Prob_Model_Z = hist_Z/(len(Model[0])/10)


#################################################
#### RANDOM WALK w LAT LINE MODEL DATA - M2 #####
#################################################

Model2 = (DATA3[:,0],DATA3[:,1],DATA3[:,2])

a = (Model2[0]/max(Model2[0]))*(68.5-2.54)+2.54
b = (Model2[1]/max(Model2[1]))*(15.24-2.54)+2.54
c = (Model2[2]/max(Model2[2]))*(15.24-2.54)+2.54

Model2 = (a,b,c)

## X DIMENSION
hist_X = np.zeros((10,len(X_BINS)-1))
Model_X = np.zeros((10,2))
for i in range(0,10):
    Model_X[i,:] = [ np.mean(Model2[0][i::10]),np.std(Model2[0][i::10]) ]
    P,bins,ignore = plt.hist(Model2[0][i::10],bins=X_BINS); plt.close()
    hist_X[i,:] = P
Prob_Model2_X = hist_X/(len(Model2[0])/10)

## Y DIMENSION
hist_Y = np.zeros((10,len(Y_BINS)-1))
Model_Y = np.zeros((10,2))
for i in range(0,10):
    Model_Y[i,:] = [ np.mean(Model2[1][i::10]),np.std(Model2[1][i::10]) ]
    P,bins,ignore = plt.hist(Model2[1][i::10],bins=Y_BINS); plt.close()
    hist_Y[i,:] = P
Prob_Model2_Y = hist_Y/(len(Model2[0])/10)
 
## Z DIMENSION
hist_Z = np.zeros((10,len(Z_BINS)-1))
Model_Z = np.zeros((10,2))
for i in range(0,10):
    Model_Z[i,:] = [ np.mean(Model2[2][i::10]),np.std(Model2[2][i::10]) ]
    P,bins,ignore = plt.hist(Model2[2][i::10],bins=Z_BINS); plt.close()
    hist_Z[i,:] = P
Prob_Model2_Z = hist_Z/(len(Model2[0])/10)

##########################################
###########BAYESIAN STATISTICS############
##########################################


Record = np.zeros((len(Real[0])/10,6))
Prob_Model1 = 0
Prob_Model2 = 0
P_M2_Loc = (1/3.)
P_M1_Loc = (1/3.)
P_M0_Loc = (1/3.)
pos = 0
TOTAL = float(len(Real[0]))
for i in range(0,int(len(Real[0])/10)): ## For the number of tadpoles.
    
    Observation = [ Real[0][(i*10):(i*10)+10],Real[1][(i*10):(i*10)+10],Real[2][(i*10):(i*10)+10] ]

    P_Model1 = 1
    P_Model2 = 1
    P_Uniform = 1
        
    for j in range(0, len(Observation)):

        P_Model1 *= float( Prob_Model_X[j][(Observation[0][j]/2.54)-1]*
                           Prob_Model_Y[j][(Observation[1][j]/2.54)-1]*
                           Prob_Model_Z[j][(Observation[2][j]/2.54)-1] )
        
        P_Model2 *= float( Prob_Model2_X[j][(Observation[0][j]/2.54)-1]*
                           Prob_Model2_Y[j][(Observation[1][j]/2.54)-1]*
                           Prob_Model2_Z[j][(Observation[2][j]/2.54)-1] )
        
        P_Uniform *= float(1/972.0)

    P_M2_Loc = P_Model2*P_M2_Loc/( P_Model1*P_M1_Loc + P_Uniform*P_M0_Loc + P_Model2*P_M2_Loc )
    P_M1_Loc = P_Model1*P_M1_Loc/( P_Model1*P_M1_Loc + P_Uniform*P_M0_Loc + P_Model2*P_M2_Loc )
    P_M0_Loc = P_Uniform*P_M0_Loc/( P_Model1*P_M1_Loc + P_Uniform*P_M0_Loc + P_Model2*P_M2_Loc )

    pos += 1
    Record[pos-1,:] = P_M2_Loc,P_M1_Loc,P_M0_Loc,P_Model2,P_Model1,P_Uniform

BF_01 = P_M0_Loc/P_M1_Loc
BF_12 = P_M1_Loc/P_M2_Loc
BF_02 = P_M0_Loc/P_M2_Loc

print 'BF_01:',BF_01,'BF_12',BF_12,'BF_02',BF_02

############################
########## PLOTS ###########
############################

#### BAYES RESULT PLOTS ####

FONTSIZE = 26
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(211)
X = np.arange(1,(len(Real[0])/10)+1)
ax.semilogy(X,Record[:,0], linewidth = 3)
ax.semilogy(X,Record[:,1], linewidth = 3)
#ax.semilogy(X,Record[:,2], linewidth = 3)
#ax.set_ylim(min([min(Record[:,0]),min(Record[:,1]),min(Record[:,2])])*10**-1, 10)
#ax.yaxis.set_major_locator(LogLocator(base=1000000000000))
plt.ylabel('P(model | data)',fontsize=FONTSIZE)
plt.xticks(fontsize = FONTSIZE)
plt.yticks(fontsize = FONTSIZE)
plt.legend(('random walk w LL model','random walk model','uniform dist model'),loc='lower left')
ax2  = fig.add_subplot(212)
ax2.semilogy(X,Record[:,3], linewidth = 3)
ax2.semilogy(X,Record[:,4], linewidth = 3)
#ax2.semilogy(X,Record[:,5], linewidth = 3)
plt.ylabel('P(data | model)',fontsize=FONTSIZE)
plt.xlabel('animal',fontsize=FONTSIZE)
plt.yticks(fontsize = FONTSIZE)
plt.xticks(fontsize = FONTSIZE)
plt.tight_layout()
plt.show()


#### DATA/MODEL PLOTS #####

FONTSIZE = 20
VMAX = 0.35
VMAX_Y = 0.6
VMAX_Z = 0.85
fig = plt.figure(figsize=(15,4.25))


#### DATA PLOTS ####

####### X DIMENSION #######

ax10 = fig.add_subplot(144)
foo = np.linspace(0,VMAX,20, endpoint=True)
colorbar = np.ones((20,2))*foo[:,np.newaxis]
ax10.imshow(colorbar, interpolation='None')
plb.setp(ax10.get_xticklabels(), visible=False)
plt.yticks(np.linspace(0,19,5),np.round_(np.linspace(0,VMAX,5,endpoint=True),2), fontsize=FONTSIZE)
plt.ylabel('P(X dimension)',fontsize=FONTSIZE)

ax = fig.add_subplot(141)
ax.imshow(Prob_Real_X,vmin = 0,vmax = VMAX,interpolation='None', aspect='auto')
plt.title('observed behavior',fontsize=FONTSIZE)
plt.xticks(np.arange(0,27,5),('up','','','','','down'), fontsize=FONTSIZE)
plt.yticks(np.arange(0,10,2),((np.arange(0,10,2)*30) + 30),fontsize=FONTSIZE)
plt.ylabel('time',fontsize=FONTSIZE)

ax4 = fig.add_subplot(142)
ax4.imshow(Prob_Model_X,vmin = 0,vmax = VMAX,interpolation='None', aspect='auto')
plt.title('random walk model',fontsize=FONTSIZE)
plt.xticks(np.arange(0,27,5),('up','','','','','down'), fontsize=FONTSIZE)
plb.setp(ax4.get_yticklabels(), visible=False)

ax7 = fig.add_subplot(143)
ax7.imshow(Prob_Model2_X,vmin = 0,vmax = VMAX,interpolation='None', aspect='auto')
plt.title('random walk w LL model',fontsize=FONTSIZE)
plt.xticks(np.arange(0,27,5),('up','','','','','down'), fontsize=FONTSIZE)
plb.setp(ax7.get_yticklabels(), visible=False)
plt.tight_layout()
plt.show()


##### Y&Z DIMENSIONS #####

##### COLOR BAR ######
fig2 = plt.figure(figsize=(15,8.5))
ax11 = fig2.add_subplot(244)
foo = np.linspace(0,VMAX_Y,20, endpoint=True)
colorbar = np.ones((20,2))*foo[:,np.newaxis]
ax11.imshow(colorbar, interpolation='None')
plb.setp(ax11.get_xticklabels(), visible=False)
plt.yticks(np.linspace(0,19,5),np.round_(np.linspace(0,VMAX_Y,5,endpoint=True),2),fontsize=FONTSIZE)
plt.ylabel('P(Y dimension)',fontsize=FONTSIZE)

ax12 = fig2.add_subplot(248)
foo = np.linspace(0,VMAX_Z,20, endpoint=True)
colorbar = np.ones((20,2))*foo[:,np.newaxis]
ax12.imshow(colorbar, interpolation='None')
plb.setp(ax12.get_xticklabels(), visible=False)
plt.yticks(np.linspace(0,19,5),np.round_(np.linspace(0,VMAX_Z,5,endpoint=True),2),fontsize=FONTSIZE)
plt.ylabel('P(Z dimension)',fontsize=FONTSIZE)

#### OBSERVED DATA #######

ax2 = fig2.add_subplot(241)
ax2.imshow(Prob_Real_Y,vmin = 0,vmax = VMAX_Y,interpolation='None', aspect='auto')
plt.title('observed behavior',fontsize=FONTSIZE)
plt.xticks(np.arange(0,6),('left','','','','','right'), fontsize=FONTSIZE)
plt.yticks(np.arange(0,10,2),((np.arange(0,10,2)*30) + 30),fontsize=FONTSIZE)
plt.ylabel('time',fontsize=FONTSIZE)

ax3 = fig2.add_subplot(245)
P_Uniform = np.ones((10,27))*(1/27.)
ax3.imshow(Prob_Real_Z,vmin = 0,vmax = VMAX_Z,interpolation='None', aspect='auto')
plt.xticks(np.arange(0,6),('bottom','','','','','surface'), fontsize=FONTSIZE)
plt.yticks(np.arange(0,10,2),((np.arange(0,10,2)*30) + 30),fontsize=FONTSIZE)
plt.ylabel('time',fontsize=FONTSIZE)


#### Random Walk Model ####



ax5 = fig2.add_subplot(242)
ax5.imshow(Prob_Model_Y,vmin = 0,vmax = VMAX_Y,interpolation='None', aspect='auto')
plt.title('random walk model',fontsize=FONTSIZE)
plt.xticks(np.arange(0,6),('left','','','','','right'), fontsize=FONTSIZE)
plb.setp(ax5.get_yticklabels(), visible=False)

ax6 = fig2.add_subplot(246)
P_Uniform = np.ones((10,27))*(1/27.)
ax6.imshow(Prob_Model_Z,vmin = 0,vmax = VMAX_Z,interpolation='None', aspect='auto')
plt.xticks(np.arange(0,6),('bottom','','','','','surface'), fontsize=FONTSIZE)
plb.setp(ax6.get_yticklabels(), visible=False)

    
#### Simple Active Walk ####


ax8 = fig2.add_subplot(243)
ax8.imshow(Prob_Model2_Y,vmin = 0,vmax = VMAX_Y,interpolation='None', aspect='auto')
plt.xticks(np.arange(0,6),('left','','','','','right'), fontsize=FONTSIZE)
plt.title('random walk w LL model',fontsize=FONTSIZE)
plb.setp(ax8.get_yticklabels(), visible=False)

ax9 = fig2.add_subplot(247)
P_Uniform = np.ones((10,27))*(1/27.)
ax9.imshow(Prob_Model2_Z,vmin = 0,vmax = VMAX_Z,interpolation='None', aspect='auto')
plt.xticks(np.arange(0,6),('bottom','','','','','surface'), fontsize=FONTSIZE)
plb.setp(ax9.get_yticklabels(), visible=False)
plt.tight_layout()
plt.show()

