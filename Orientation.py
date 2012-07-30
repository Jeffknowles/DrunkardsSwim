import matplotlib.pyplot as plt
import numpy as np

FLOW = 1

if FLOW == 0:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Gent/4cm_NoFlow_Gent.csv', delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_NOLLNoFlow.csv', delimiter=',')
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_LLNoFlow.csv', delimiter=',')
if FLOW == 1:
    DATA1 = np.genfromtxt('/Users/brianschmidt/Dropbox/Data_for_Brian/Brian_Analysis_Files/Gent/4cm_Flow_Gent.csv',delimiter=',')
    DATA2 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_NoLLFlow.csv',delimiter=',')    
    DATA3 = np.genfromtxt('/Users/brianschmidt/GDrive/Tadpole/randomwalk/DrunkardsSwim/Model_NEW4cm_LLFlow.csv', delimiter=',')

Real = DATA1[:,6]
Model1 = DATA2[:,3]
Model2 = DATA3[:,3]

BINS = np.linspace(0,360,360/15.0) ## 15degree bins

Real_Hist,b,w = plt.hist(Real,bins=BINS)
Real_Hist = np.append(Real_Hist,Real_Hist[0])

Model1_Hist,b,w = plt.hist(Model1,bins=BINS)
Model1_Hist = np.append(Model1_Hist,Model1_Hist[0])

Model2_Hist,b,w = plt.hist(Model2,bins=BINS)
Model2_Hist = np.append(Model2_Hist,Model2_Hist[0])



RAD_BINS = BINS/(180./np.pi); plt.close()
## Data Plot
plt.rc('grid', color='gray', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=20)
width, height = plt.rcParams['figure.figsize']
size = min(width, height)
fig = plt.figure(figsize=(size, size))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='w')
plt.polar(RAD_BINS,Real_Hist/float(sum(Real_Hist)), 'k',linewidth=4, marker='o')
ax.set_rmax(0.2)
plt.show()

# Model1 Plot
plt.rc('grid', color='gray', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=20)
width, height = plt.rcParams['figure.figsize']
size = min(width, height)
fig = plt.figure(figsize=(size, size))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='w')
plt.polar(RAD_BINS,Model1_Hist/float(sum(Model1_Hist)), 'k',linewidth=4, marker='o')
ax.set_rmax(0.2)
plt.show()

# Model2 Plot
plt.rc('grid', color='gray', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=25)
plt.rc('ytick', labelsize=20)
width, height = plt.rcParams['figure.figsize']
size = min(width, height)
fig = plt.figure(figsize=(size, size))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True, axisbg='w')
plt.polar(RAD_BINS,Model2_Hist/float(sum(Model2_Hist)), 'k',linewidth=4, marker='o')
ax.set_rmax(0.2)
plt.show()

