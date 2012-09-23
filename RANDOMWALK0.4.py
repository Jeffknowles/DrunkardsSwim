import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import walkfunction as wlk


##### Options #####
###################

numfrog = 10 # Number of tadpoles you would like to simulate.
TIME = 300 # This is the time of the simulation in seconds for each flow condition.
FLOW_DYNAMICS = 'jeb' ## 'jeb' or 'new'
binsize = 0.1

# set the flow and lat line conditions
flowspeed = np.array([10])#, 10, 8, 8, 6, 6])
latlin =    np.array([ 0])#,  1, 0, 1, 0, 1])

#### end options #####
######################

lat = ['llOFF', 'llON'] # this is just for naming files.
froglen = TIME/binsize

for i in range(0,len(flowspeed)):
    
    # set up run conditions and allocate memory.
    LATLINE = latlin[i] # 0=off, 1=on
    FLOWSPEED = flowspeed[i] ## in cm/s

    frogData_NF = {'time': np.zeros((TIME/binsize*numfrog)),
                'track': np.zeros((TIME/binsize*numfrog,3)),
                'velocity':np.zeros((TIME/binsize*numfrog,3)),
                'orientation':np.zeros((TIME/binsize*numfrog)),
                'lateral_line':np.zeros((TIME/binsize*numfrog)),
                'state':np.zeros((TIME/binsize*numfrog))
                }

    frogData_F = {'time': np.zeros((TIME/binsize*numfrog)),
                'track': np.zeros((TIME/binsize*numfrog,3)),
                'velocity':np.zeros((TIME/binsize*numfrog,3)),
                'orientation':np.zeros((TIME/binsize*numfrog)),
                'lateral_line':np.zeros((TIME/binsize*numfrog)),
                'state':np.zeros((TIME/binsize*numfrog))
                }
    
    for j in range(0,numfrog):
        
        #(SIMLENGTH,BINSIZE,FLOW_SPEED,IPOS,LATLINE,VERSION)  - Version refers to flow dynamics.
        run_data_NF = wlk.randwalk(TIME,binsize,0, np.array([0.34, 0.075, 0.15]),latline = LATLINE,flow_version = FLOW_DYNAMICS)

        # there is likely a more elegant way to pack these data up.
        frogData_NF['time'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_NF['time']
        frogData_NF['track'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_NF['track']
        frogData_NF['velocity'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_NF['velocity']
        frogData_NF['orientation'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_NF['orientation']
        frogData_NF['lateral_line'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_NF['lateral_line']
        frogData_NF['state'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_NF['state']

        frogEndPoint = frogData_NF['track'][-1:,:]
        run_data_F = wlk.randwalk(TIME,binsize, FLOWSPEED, frogEndPoint,latline = LATLINE,flow_version = FLOW_DYNAMICS)

        
        frogData_F['time'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_F['time']
        frogData_F['track'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_F['track']
        frogData_F['velocity'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_F['velocity']
        frogData_F['orientation'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_F['orientation']
        frogData_F['lateral_line'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_F['lateral_line']
        frogData_F['state'][(j+1)*froglen-froglen:(j+1)*froglen] = run_data_F['state']


    # saves files as .npz files.  Use 'load()' to load files for analysis.
    time = frogData_NF['time'][:,np.newaxis] # turn the array 90 deg for easier usage with Analysis Functions.
    track = frogData_NF['track']
    velocity = frogData_NF['velocity']
    orientation = frogData_NF['orientation'][:,np.newaxis]
    lateral_line = frogData_NF['lateral_line'][:,np.newaxis]
    state = frogData_NF['state'][:,np.newaxis]

    # apparently only accepts 6 variables max. Look into savez_compressed for storing smaller files.
    np.savez(str(FLOW_DYNAMICS.upper()) + '_' + str(FLOWSPEED) + 'cm_' + str(lat[LATLINE]) + '_NoFlow',
             time=time, track=track, velocity=velocity, orientation=orientation, lateral_line=lateral_line,
             state=state)

    time = frogData_F['time'][:,np.newaxis]
    track = frogData_F['track']
    velocity = frogData_F['velocity']
    orientation = frogData_F['orientation'][:,np.newaxis]
    lateral_line = frogData_F['lateral_line'][:,np.newaxis]
    state = frogData_F['state'][:,np.newaxis]
    
    np.savez(str(FLOW_DYNAMICS.upper()) + '_' + str(FLOWSPEED) + 'cm_' + str(lat[LATLINE]) + '_Flow',
             time=time, track=track, velocity=velocity, orientation=orientation, lateral_line=lateral_line,
             state=state)

