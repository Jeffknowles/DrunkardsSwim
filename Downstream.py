import matplotlib.pyplot as plt
import numpy as np

FLOW = 1

if FLOW == 0:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/2cm_NoFlow_Control.csv', delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_NOLLNoFlow.csv', delimiter=',')
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_LLNoFlow.csv', delimiter=',')
if FLOW == 1:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/2cm_Flow_Control.csv',delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_NoLLFlow.csv',delimiter=',')    
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW2cm_LLFlow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>34.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>34.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>34.0)/float(len(Model2[i::10]))

plt.plot(np.arange(30,301,30),Real_Downstream,'k-v',linewidth=3, markersize=10)
plt.plot(np.arange(30,301,30),Model1_Downstream,'g-v',linewidth=3, markersize=10)
plt.plot(np.arange(30,301,30),Model2_Downstream,'r-v',linewidth=3, markersize=10)
plt.show()



if FLOW == 0:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/4cm_NoFlow_Control.csv', delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_NOLLNoFlow.csv', delimiter=',')
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_LLNoFlow.csv', delimiter=',')
if FLOW == 1:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/4cm_Flow_Control.csv',delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_NoLLFlow.csv',delimiter=',')    
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_LLFlow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>34.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>34.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>34.0)/float(len(Model2[i::10]))

plt.plot(np.arange(30,301,30),Real_Downstream,'k-.',linewidth=3)
plt.plot(np.arange(30,301,30),Model1_Downstream,'g-.',linewidth=3)
plt.plot(np.arange(30,301,30),Model2_Downstream,'r-.',linewidth=3)
plt.show()


if FLOW == 0:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/6cm_NoFlow_Control.csv', delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_NOLLNoFlow.csv', delimiter=',')
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_LLNoFlow.csv', delimiter=',')
if FLOW == 1:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/6cm_Flow_Control.csv',delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_NoLLFlow.csv',delimiter=',')    
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW6cm_LLFlow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>34.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>34.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>34.0)/float(len(Model2[i::10]))

plt.plot(np.arange(30,301,30),Real_Downstream,'k-',linewidth=3)
plt.plot(np.arange(30,301,30),Model1_Downstream,'g-',linewidth=3)
plt.plot(np.arange(30,301,30),Model2_Downstream,'r-',linewidth=3)
plt.show()



if FLOW == 0:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/8cm_NoFlow_Control.csv', delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_NOLLNoFlow.csv', delimiter=',')
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_LLNoFlow.csv', delimiter=',')
if FLOW == 1:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/8cm_Flow_Control.csv',delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_NoLLFlow.csv',delimiter=',')    
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW8cm_LLFlow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>34.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>34.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>34.0)/float(len(Model2[i::10]))

plt.plot(np.arange(30,301,30),Real_Downstream,'k:',linewidth=3)
plt.plot(np.arange(30,301,30),Model1_Downstream,'g:',linewidth=3)
plt.plot(np.arange(30,301,30),Model2_Downstream,'r:',linewidth=3)
plt.show()





if FLOW == 0:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/10cm_NoFlow_Control.csv', delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_NOLLNoFlow.csv', delimiter=',')
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_LLNoFlow.csv', delimiter=',')
if FLOW == 1:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Control/10cm_Flow_Control.csv',delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_NoLLFlow.csv',delimiter=',')    
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW10cm_LLFlow.csv', delimiter=',')

Real = DATA1[:,4]
Model1 = DATA2[:,0]
Model2 = DATA3[:,0]

Real_Downstream = np.zeros(10)
Model1_Downstream = np.zeros(10)
Model2_Downstream = np.zeros(10)
for i in range(0,10):
    Real_Downstream[i] = np.sum(Real[i::10]>34.0)/float(len(Real[i::10]))
    Model1_Downstream[i] = np.sum(Model1[i::10]>34.0)/float(len(Model1[i::10]))
    Model2_Downstream[i] = np.sum(Model2[i::10]>34.0)/float(len(Model2[i::10]))

plt.plot(np.arange(30,301,30),Real_Downstream,'k--',linewidth=3)
plt.plot(np.arange(30,301,30),Model1_Downstream,'g--',linewidth=3)
plt.plot(np.arange(30,301,30),Model2_Downstream,'r--',linewidth=3)
plt.ylabel('proportion downstream',fontsize=25)
plt.xlabel('time (s)',fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim([30,300])
plt.ylim([0.5,1.01])
plt.tight_layout()
plt.show()


