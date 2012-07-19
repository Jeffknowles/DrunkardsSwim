import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import walkfunction as wlk

############################
##### USER INPUT BELOW #####
############################

NUMFROG = 2 # Number of tadpoles you would like to simulate.
TIME = 300 # This is the time of the simulation in seconds for each flow condition. 
NO_FLOW_CSV_FILE_NAME ='Test.csv'
FLOW_CSV_FILE_NAME = 'Test.csv'
LATLINE = 1 # 0=off, 1=on
FLOWSPEED = 8 ## in cm/s, has to be 0, 6, 8 or 10cm/s
FLOW_DYNAMICS = 'New' ## 'JEB' or 'NEW'

##############################
####### END USER INPUT #######
##############################

# This begins the simulation of 0cm/s flow conditions.
FROGDATA = np.zeros((NUMFROG*10,4))
FINALPOSITION = np.zeros((NUMFROG,4))

for j in range(0,NUMFROG):
    #(SIMLENGTH,BINSIZE,FLOW_SPEED,IPOS,LATLINE,VERSION)  - Version refers to flow dynamics.
    frog,orient = wlk.randwalk(300,0.1, 0, np.array([0.37, 0.075, 0.15]),LATLINE,FLOW_DYNAMICS)

    frogData = np.zeros((10,4))
    # Here we are extracting only the position at 30 second intervals.
    for i in range(0,10):
        Data = frog[(i*300)+299,:]
        Orient = orient[(i*300)+299]
        frogData[i,:] = np.append(Data,Orient)
        
    FINALPOSITION[j,:] = frogData[9,:]
    FROGDATA[(j+1)*10-10:(j+1)*10,:] = frogData
    
FROGDATA[:,0:3] = FROGDATA[:,0:3]*100 #convert meters into cm

np.savetxt(NO_FLOW_CSV_FILE_NAME, FROGDATA, delimiter=',')

## This begins the FLOW simulation: 

FROGDATA = np.zeros((NUMFROG*10,4))

for j in range(0,NUMFROG):

    ipos = FINALPOSITION[j,:] # this indicates initial position which is the last recorded
    #position for each tadpole under 0cm/s, thus creating a continuous
    #trial, just like the pulsed flow experiments we run.
    frog,orient = wlk.randwalk(300,0.1, FLOWSPEED, ipos[0:3],LATLINE,FLOW_DYNAMICS)

    frogData = np.zeros((10,4))

    for i in range(0,10):

        Data = frog[(i*300)+299,:]
        Orient = orient[(i*300)+299]
        frogData[i,:] = np.append(Data,Orient)
    
    FROGDATA[(j+1)*10-10:(j+1)*10,:] = frogData

FROGDATA[:,0:3] = FROGDATA[:,0:3]*100

np.savetxt(FLOW_CSV_FILE_NAME, FROGDATA, delimiter=',')

