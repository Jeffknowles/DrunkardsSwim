import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import walkfunction as wlk

############################
##### USER INPUT BELOW #####
############################

NUMFROG = 5 # Number of tadpoles you would like to simulate.
TIME = 300 # This is the time of the simulation in seconds for each flow condition.
FLOW_DYNAMICS = 'NEW' ## 'JEB' or 'NEW'

NO_FLOW = np.array(['Model_NEW10cm_NoLL_NoFlow.csv', 'Model_NEW10cm_LL_NoFlow.csv',
                    'Model_NEW8cm_NoLL_NoFlow.csv', 'Model_NEW8cm_LL_NoFlow.csv',
                    'Model_NEW6cm_NoLL_NoFlow.csv', 'Model_NEW6cm_LL_NoFlow.csv',
                    'Model_NEW4cm_NoLL_NoFlow.csv', 'Model_NEW4cm_LL_NoFlow.csv',
                    'Model_NEW2cm_NoLL_NoFlow.csv', 'Model_NEW2cm_LL_NoFlow.csv'])

FLOW = np.array(['Model_NEW10cm_NoLL_Flow.csv', 'Model_NEW10cm_LL_Flow.csv',
                 'Model_NEW8cm_NoLL_Flow.csv', 'Model_NEW8cm_LL_Flow.csv',
                 'Model_NEW6cm_NoLL_Flow.csv', 'Model_NEW6cm_LL_Flow.csv',
                 'Model_NEW4cm_NoLL_Flow.csv', 'Model_NEW4cm_LL_Flow.csv',
                 'Model_NEW2cm_NoLL_Flow.csv', 'Model_NEW2cm_LL_Flow.csv'])

flowspeed = np.array([10,10,8,8,6,6,4,4,2,2])
latlin =    np.array([0,1,0,1,0,1,0,1,0,1])


for i in range(0,len(flowspeed)):
    NO_FLOW_CSV_FILE_NAME = NO_FLOW[i]
    FLOW_CSV_FILE_NAME = FLOW[i]
    LATLINE = latlin[i] # 0=off, 1=on
    FLOWSPEED = flowspeed[i] ## in cm/s, has to be 0, 6, 8 or 10cm/s


##############################
####### END USER INPUT #######
##############################

# This begins the simulation of 0cm/s flow conditions.
    FROGDATA = np.zeros((NUMFROG*10,4))
    FINALPOSITION = np.zeros((NUMFROG,4))

    for j in range(0,NUMFROG):
        #(SIMLENGTH,BINSIZE,FLOW_SPEED,IPOS,LATLINE,VERSION)  - Version refers to flow dynamics.
        run_data = wlk.randwalk(300,0.1, 0, np.array([0.37, 0.075, 0.15]),latline = LATLINE,flow_version = FLOW_DYNAMICS)
        frogData = np.zeros((10,4))
    # Here we are extracting only the position at 30 second intervals.
        for i in range(0,10):
            Data = run_data['track'][(i*300)+299,:]
            Orient = run_data['orientation'][(i*300)+299]
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
        run_data = wlk.randwalk(300,0.1, FLOWSPEED, ipos[0:3],LATLINE,FLOW_DYNAMICS)
        frogData = np.zeros((10,4))

        for i in range(0,10):
            Data = run_data['track'][(i*300)+299,:]
            Orient = run_data['orientation'][(i*300)+299]
            frogData[i,:] = np.append(Data,Orient)
    
        FROGDATA[(j+1)*10-10:(j+1)*10,:] = frogData

    FROGDATA[:,0:3] = FROGDATA[:,0:3]*100

    np.savetxt(FLOW_CSV_FILE_NAME, FROGDATA, delimiter=',')

