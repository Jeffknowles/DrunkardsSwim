import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import walkfunction as wlk

############################
##### USER INPUT BELOW #####
############################

NUMFROG = 2000 # Number of tadpoles you would like to simulate.
TIME = 300 # This is the time of the simulation in seconds for each flow condition. 
NO_FLOW_CSV_FILE_NAME ='FrogModelLL8cm.csv'
FLOW_CSV_FILE_NAME = 'FrogModelFlowLL8cm.csv'
LATLINE = 1 # 0=off, 1=on
FLOWSPEED = 8 ## in cm/s, has to be 0, 6, 8 or 10cm/s

##############################
####### END USER INPUT #######
##############################

# This begins the simulation of 0cm/s flow conditions.
FROGDATA = np.zeros((NUMFROG*10,3))
FINALPOSITION = np.zeros((NUMFROG,3))

for j in range(0,NUMFROG):
    
    frog = wlk.randwalk(300,0.1, 0, np.array([0.37, 0.075, 0.15]),LATLINE)

    frogData = np.zeros((10,3))
    # Here we are extracting only the position at 30 second intervals.
    for i in range(0,10):
        Data = frog[(i*300)+299,:]
    
        frogData[i,:] = Data
    
    FINALPOSITION[j,:] = frogData[9,:]
    FROGDATA[(j+1)*10-10:(j+1)*10,:] = frogData
    
FROGDATA = FROGDATA*100 #convert meters into cm

np.savetxt(NO_FLOW_CSV_FILE_NAME, FROGDATA, delimiter=',')

## This begins the FLOW simulation: 

FROGDATA = np.zeros((NUMFROG*10,3))

for j in range(0,NUMFROG):

    ipos = FINALPOSITION[j,:] # this indicates initial position which is the last recorded
    #position for each tadpole under 0cm/s, thus creating a continuous
    #trial, just like the pulsed flow experiments we run.
    frog = wlk.randwalk(300,0.1, FLOWSPEED, ipos,LATLINE)

    frogData = np.zeros((10,3))

    for i in range(0,10):
        Data = frog[(i*300)+299,:]
    
        frogData[i,:] = Data
    
    FROGDATA[(j+1)*10-10:(j+1)*10,:] = frogData

FROGDATA = FROGDATA*100

np.savetxt(FLOW_CSV_FILE_NAME, FROGDATA, delimiter=',')
