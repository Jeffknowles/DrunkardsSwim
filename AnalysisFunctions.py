import matplotlib.pylab as plt
import numpy as np
import scipy as sp

def KL_Diverg(Real_Data, Model_Data, Dim=0, BIN_RESOLUTION=75,PRINT=0):
    """
    Computes the Kullback-Leibler Divergence between two probability distributions
    KL_Diverg = Sum( P * log2( P/ Model) )

    for more info,
    see: http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    and: http://www.snl.salk.edu/~shlens/kl.pdf 
    
    Input
    -----
    Real_Data: The distribution you are fitting to
    Model_Data: The model you are testing

    Output
    ------
    KL_Divergence: in bits (log2). Single float.

    Options
    -------
    Dim: the dimension of data you are look at, default = 0
    BIN_RESOLUTION: the bin size for probability calculation, default = 75
    PRINT: option of printing the results to the console, deflault = 0, 1 will print
    """
    
    P_Data = np.zeros(len(Real_Data[:,Dim])/10.)
    for i in range(0,int(len(Real_Data[:,Dim])/10.)):
        P_Data[i] = np.var(Real_Data[:,Dim][10.*i:(10.*i)+10.])

    P_Model = np.zeros(len(Model_Data[:,Dim])/10.)
    for i in range(0,int(len(Model_Data[:,Dim])/10.)):
        P_Model[i] = np.var(Model_Data[:,Dim][10.*i:(10.*i)+10.])


    ## Compute Probabilities

    BINS = np.linspace(min( min(P_Data),min(P_Model) ),
                        max (max(P_Data),max(P_Model)),BIN_RESOLUTION)

    # find distributions
    Prob_Data,q = np.histogram(P_Data,bins=BINS)
    Prob_Model,q = np.histogram(P_Model,bins=BINS)

    # prevent zeros in division 
    Prob_Data += 1.
    Prob_Model += 1.

    # find probabilities
    Prob_Data = np.true_divide(Prob_Data,sum(Prob_Data))
    Prob_Model = np.true_divide(Prob_Model,sum(Prob_Model))

    # compute KL-Divergence between Model and Observed Distributions.
    KLdiverg_DataVsModel = 0

    for i in range(0,len(BINS)-1):
        KLdiverg_DataVsModel += Prob_Data[i]*np.log2(Prob_Data[i]/Prob_Model[i])

    if PRINT == 1:
        print 'KL-diverg = ', KLdiverg_DataVsModel

    return KLdiverg_DataVsModel



def ProportionDown(Data,Loc = 34.0,Dim = 0,TimePoints = 10):
    """
    Compute the proportion of animals downstream

    Input
    -----
    Data: Position of animals

    Output
    ------
    Downstream: Proportion of animals downstream at each Time Point

    Options
    -------
    Loc: location of downstream cut off. Default = 34.0 (half way in the tank)
    Dim: dimension of the data to use. Default = 0 (usually where X data is stored)
    TimePoints: number of time points observed for each animal. Defaul = 10 (30, 60, 90, etc.)
    """

    Data = Data[:,Dim]

    Downstream = np.zeros(TimePoints)

    for i in range(0,TimePoints):
        Downstream[i] = np.sum(Data[i::TimePoints]>Loc)/float(len(Data[i::TimePoints]))

    return Downstream



def OrientationHist(Data,Dim = 3,Bin_Resolution = 15):
    """
    Compute an orientaiton histogram

    Input
    -----
    Data: Data file with orientation measurements

    Output
    ------
    Hist: count of data in each bin
    BINS: edges of each bin used in histogram

    Options
    -------
    Dim: dimension of the data to use. Default = 3 (usually where orientation data is stored)
    Bin_Resolution: size of the bins used in histogram. Default = 15 
    
    """
    Data = Data[:,3]

    BINS = np.linspace(0,360,360/Bin_Resolution) ## 15degree bins

    Hist,b = np.histogram(Data,bins=BINS)
    Hist = np.append(Hist,Hist[0])

    return Hist, BINS


def OrientationPlot(OrientationHist_Data,BINS):
    """
    Plot an orienation histogram. For use with OrientationHist function

    Input
    -----
    OrientationHist_Data: computed from OrienationHist function
    BINS: bins used to compute OrientationHist

    Output
    ------
    Produces a polar plot with the orientation histogram data

    """
    RAD_BINS = BINS/(180./np.pi)
    ## Data Plot
    plt.rc('grid', color='gray', linewidth=1, linestyle='-')
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=20)
    width, height = plt.rcParams['figure.figsize']
    size = min(width, height)
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='w')
    plt.polar(RAD_BINS,OrientationHist_Data/float(sum(OrientationHist_Data)), 'k',linewidth=4, marker='o')
    ax.set_rmax(0.2)
    plt.show()







###################### 
### Bayes Analysis ###
######################

def StandardizeData(Data, Loc = np.array([0,1,2]) ):
    """
    Make sure that data is all in the same format for analyses
    Input
    -----
    Data: data file

    Output
    ------
    Standard_Data: data in standardized format where; X = [1,68.5], Y = [1,15.2], Z = [1,15.2]

    Options
    -------
    Loc: locations of X,Y,Z data in an array. Default = np.array([0,1,2])
    

    """
    X = Data[:,Loc[0]]
    Y = Data[:,Loc[1]]
    Z = Data[:,Loc[2]]
    

    # Find proper scale, i.e. Inches or Cm
    test_X = max(X)
    test_YZ = max(max(Y),max(Z))
    
    if test_X > 68.6:
        print 'Error!! X data should not exceed 68.5 cm'
    elif test_YZ > 16.0:
        print 'Error!! Y and Z data should not exceed 15.6 cm'
    else:
        if test_X > 30:
            Max_X = 68.0
            Max_YZ = 16.0
        else:
            Max_X = 27.0
            Max_YZ = 6.0
    
    # Make sure that min = 1.0
    X = X + (1.0 - min(X))
    Y = Y + (1.0 - min(Y))
    Z = Z + (1.0 - min(Z))


    
    # Now scale to proper max for each dimension
    
    X = X/Max_X*68.5
    Y = Y/Max_YZ*15.2
    Z = Z/Max_YZ*15.2
    
    Standard_Data = np.array([X.T, Y.T, Z.T])
    
    return Standard_Data

def MarginalProbabilityDist(Data)
    """
    Under development

    Need to set up marginal distributions

    """

    ## np.histogram Bins
    X_BINS = np.linspace(1,68.58,27.0))
    Y_BINS = np.linspace(1,15.24,6.0))
    Z_BINS = np.linspace(1,15.24,6.0))
    
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



def BayesHypothesisTest(Observed_Data,Model1_Data,Model2_Data)
    """
    Under development
    """

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

    return BF_01,BF_02,BF_12









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

