import matplotlib.pyplot as plt
import numpy as np


DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/2cm_NoFlow_Control.csv', delimiter=',')
DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_NOLL_NoFlow.csv', delimiter=',')
DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_LL_NoFlow.csv', delimiter=',')

DATA4 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/2cm_Flow_Control.csv',delimiter=',')
DATA5 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_NoLL_Flow.csv',delimiter=',')    
DATA6 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_LL_Flow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]
Real_Flow = DATA4[:,4]
Model1_Flow = DATA5[:,0]
Model2_Flow = DATA6[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
Real_Downstream_Flow = np.zeros(10)
Model1_Downstream_Flow = np.zeros(10)
Model2_Downstream_Flow = np.zeros(10)

for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>64.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>64.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>64.0)/float(len(Model2[i::10]))
    Real_Downstream_Flow[i] = np.sum(Real_Flow[i::10]>64.0)/float(len(Real_Flow[i::10]))
    Model1_Downstream_Flow[i] = np.sum(Model1_Flow[i::10]>64.0)/float(len(Model1_Flow[i::10]))
    Model2_Downstream_Flow[i] = np.sum(Model2_Flow[i::10]>64.0)/float(len(Model2_Flow[i::10]))

Real_Downstream = np.array([Real_Downstream,Real_Downstream_Flow]).flatten()
Model1_Downstream = np.array([Model1_Downstream,Model1_Downstream_Flow]).flatten()
Model2_Downstream = np.array([Model2_Downstream,Model2_Downstream_Flow]).flatten()

plt.plot(np.arange(30,601,30),Real_Downstream,'k-v',linewidth=3, markersize=10)
plt.plot(np.arange(30,601,30),Model1_Downstream,'g-v',linewidth=3, markersize=10)
plt.plot(np.arange(30,601,30),Model2_Downstream,'r-v',linewidth=3, markersize=10)
plt.show()




DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/4cm_NoFlow_Control.csv', delimiter=',')
DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_NOLL_NoFlow.csv', delimiter=',')
DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_LL_NoFlow.csv', delimiter=',')

DATA4 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/4cm_Flow_Control.csv',delimiter=',')
DATA5 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_NoLL_Flow.csv',delimiter=',')    
DATA6 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_LL_Flow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]
Real_Flow = DATA4[:,4]
Model1_Flow = DATA5[:,0]
Model2_Flow = DATA6[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
Real_Downstream_Flow = np.zeros(10)
Model1_Downstream_Flow = np.zeros(10)
Model2_Downstream_Flow = np.zeros(10)

for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>64.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>64.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>64.0)/float(len(Model2[i::10]))
    Real_Downstream_Flow[i] = np.sum(Real_Flow[i::10]>64.0)/float(len(Real_Flow[i::10]))
    Model1_Downstream_Flow[i] = np.sum(Model1_Flow[i::10]>64.0)/float(len(Model1_Flow[i::10]))
    Model2_Downstream_Flow[i] = np.sum(Model2_Flow[i::10]>64.0)/float(len(Model2_Flow[i::10]))

Real_Downstream = np.array([Real_Downstream,Real_Downstream_Flow]).flatten()
Model1_Downstream = np.array([Model1_Downstream,Model1_Downstream_Flow]).flatten()
Model2_Downstream = np.array([Model2_Downstream,Model2_Downstream_Flow]).flatten()
plt.plot(np.arange(30,601,30),Real_Downstream,'k-.',linewidth=3)
plt.plot(np.arange(30,601,30),Model1_Downstream,'g-.',linewidth=3)
plt.plot(np.arange(30,601,30),Model2_Downstream,'r-.',linewidth=3)
plt.show()



DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/6cm_NoFlow_Control.csv', delimiter=',')
DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_NOLL_NoFlow.csv', delimiter=',')
DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_LL_NoFlow.csv', delimiter=',')

DATA4 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/6cm_Flow_Control.csv',delimiter=',')
DATA5 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_NoLL_Flow.csv',delimiter=',')    
DATA6 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_LL_Flow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]
Real_Flow = DATA4[:,4]
Model1_Flow = DATA5[:,0]
Model2_Flow = DATA6[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
Real_Downstream_Flow = np.zeros(10)
Model1_Downstream_Flow = np.zeros(10)
Model2_Downstream_Flow = np.zeros(10)

for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>64.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>64.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>64.0)/float(len(Model2[i::10]))
    Real_Downstream_Flow[i] = np.sum(Real_Flow[i::10]>64.0)/float(len(Real_Flow[i::10]))
    Model1_Downstream_Flow[i] = np.sum(Model1_Flow[i::10]>64.0)/float(len(Model1_Flow[i::10]))
    Model2_Downstream_Flow[i] = np.sum(Model2_Flow[i::10]>64.0)/float(len(Model2_Flow[i::10]))

Real_Downstream = np.array([Real_Downstream,Real_Downstream_Flow]).flatten()
Model1_Downstream = np.array([Model1_Downstream,Model1_Downstream_Flow]).flatten()
Model2_Downstream = np.array([Model2_Downstream,Model2_Downstream_Flow]).flatten()
plt.plot(np.arange(30,601,30),Real_Downstream,'k-',linewidth=3)
plt.plot(np.arange(30,601,30),Model1_Downstream,'g-',linewidth=3)
plt.plot(np.arange(30,601,30),Model2_Downstream,'r-',linewidth=3)
plt.show()




DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/8cm_NoFlow_Control.csv', delimiter=',')
DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_NOLL_NoFlow.csv', delimiter=',')
DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_LL_NoFlow.csv', delimiter=',')

DATA4 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/8cm_Flow_Control.csv',delimiter=',')
DATA5 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_NoLL_Flow.csv',delimiter=',')    
DATA6 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_LL_Flow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]
Real_Flow = DATA4[:,4]
Model1_Flow = DATA5[:,0]
Model2_Flow = DATA6[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
Real_Downstream_Flow = np.zeros(10)
Model1_Downstream_Flow = np.zeros(10)
Model2_Downstream_Flow = np.zeros(10)

for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>64.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>64.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>64.0)/float(len(Model2[i::10]))
    Real_Downstream_Flow[i] = np.sum(Real_Flow[i::10]>64.0)/float(len(Real_Flow[i::10]))
    Model1_Downstream_Flow[i] = np.sum(Model1_Flow[i::10]>64.0)/float(len(Model1_Flow[i::10]))
    Model2_Downstream_Flow[i] = np.sum(Model2_Flow[i::10]>64.0)/float(len(Model2_Flow[i::10]))

Real_Downstream = np.array([Real_Downstream,Real_Downstream_Flow]).flatten()
Model1_Downstream = np.array([Model1_Downstream,Model1_Downstream_Flow]).flatten()
Model2_Downstream = np.array([Model2_Downstream,Model2_Downstream_Flow]).flatten()
plt.plot(np.arange(30,601,30),Real_Downstream,'k:',linewidth=3)
plt.plot(np.arange(30,601,30),Model1_Downstream,'g:',linewidth=3)
plt.plot(np.arange(30,601,30),Model2_Downstream,'r:',linewidth=3)
plt.show()





DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/10cm_NoFlow_Control.csv', delimiter=',')
DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_NOLL_NoFlow.csv', delimiter=',')
DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_LL_NoFlow.csv', delimiter=',')

DATA4 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/10cm_Flow_Control.csv',delimiter=',')
DATA5 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_NoLL_Flow.csv',delimiter=',')    
DATA6 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_LL_Flow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]
Real_Flow = DATA4[:,4]
Model1_Flow = DATA5[:,0]
Model2_Flow = DATA6[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
Real_Downstream_Flow = np.zeros(10)
Model1_Downstream_Flow = np.zeros(10)
Model2_Downstream_Flow = np.zeros(10)

for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>64.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>64.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>64.0)/float(len(Model2[i::10]))
    Real_Downstream_Flow[i] = np.sum(Real_Flow[i::10]>64.0)/float(len(Real_Flow[i::10]))
    Model1_Downstream_Flow[i] = np.sum(Model1_Flow[i::10]>64.0)/float(len(Model1_Flow[i::10]))
    Model2_Downstream_Flow[i] = np.sum(Model2_Flow[i::10]>64.0)/float(len(Model2_Flow[i::10]))

Real_Downstream = np.array([Real_Downstream,Real_Downstream_Flow]).flatten()
Model1_Downstream = np.array([Model1_Downstream,Model1_Downstream_Flow]).flatten()
Model2_Downstream = np.array([Model2_Downstream,Model2_Downstream_Flow]).flatten()

plt.plot(np.arange(30,601,30),Real_Downstream,'k--',linewidth=3)
plt.plot(np.arange(30,601,30),Model1_Downstream,'g--',linewidth=3)
plt.plot(np.arange(30,601,30),Model2_Downstream,'r--',linewidth=3)
plt.plot(np.ones(100)*300,np.arange(0,1,0.01),'k--',linewidth = 2)
plt.ylabel('proportion downstream wall',fontsize=25)
plt.xlabel('time (s)',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim([30,600])
plt.ylim([0.0,1.0])
plt.tight_layout()
plt.show()


