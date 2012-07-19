import numpy as np
import scipy as sp

#####################
#### SUBFUNCTION ####
#####################

def randpick(TRANSMAT):
	choice = np.random.random_integers(1,1000,1)
	pick = 0 if choice[0] <= TRANSMAT[0]*1000 else 1
	return pick

def orient_angle(x,y): ## relative to flow, away from flow = 180
	if x >= 0 and y >= 0:
		angle = np.arctan(x/y)*180/np.pi
	if x >= 0 and y < 0:
		angle = ((2*np.pi+np.arctan(x/y))*180/np.pi)-90
	if x < 0 and y >= 0:
		angle = (2*np.pi+np.arctan(x/y))*180/np.pi
	if x < 0 and y < 0:
		angle = (np.arctan(x/y)*180/np.pi)+90
	return angle

####################################
########### MAIN FUNCTION ##########
####################################
def randwalk(SIMLENGTH,BINSIZE,FLOW_SPEED,IPOS,LATLINE,VERSION):

    ## set number of dimensions
    NDIM = 3

    ## define animal's size and drag (according to Liu et al 1997)
    AREA = 2e-4   ## set the square area
    Cdrag = 0.0287 ## from paper
    MASS = 0.0004

    ########################## Turning Parameters
    TURNINGDIST = 'normal' ## select distribution from which to draw turns
    LAMBDA = 1 ##set average length of time between turns
    SDA_T = 0.1*np.pi ##standard deviation of turning angles for normal turning dist
    SDA_P = 0.1*np.pi
    AMT_T = np.pi  ##Angle of Maximum Turn for uniform turning distributon
    AMT_P = np.pi

    THETA = 0 ## set initial orientation angles
    PHI = 0

    ########################## Set boundries
    BOUNDTEST = 1        ## Turn boundries on or off (1 or 0)
    MINS = np.array([0, 0, 0])       ## minimum boundries
    MAXS = np.array([0.685, 0.15, 0.15])## maximum boundries

    ########################## Set Gravity (deps on bouyancy)
    GRAV = np.array([0, 0, -.05])

    ########################## Set Flow
    if FLOW_SPEED is 0:
	FLOW_ON = 0
    elif FLOW_SPEED > 0:
	FLOW_ON = 1
	
	if VERSION == 'JEB':
		CURRENT_X = np.array([[ 0.08,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.08],
		      [ 0.15 ,  0.3 ,  0.3 ,  0.5 ,  0.3 ,  0.15 ],
		      [ 0.15 ,  0.8 ,  0.8 ,  0.8 ,  0.5 ,  0.15 ],
		      [ 0.15 ,  1.0 ,  1.0 ,  0.6 ,  1.0 ,  0.15 ],
		      [ 0.15 ,  0.7 ,  1.0 ,  1.0 ,  0.8 ,  0.15 ],
		      [ 0.10 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ]])
		CURRENT_X = CURRENT_X*FLOW_SPEED*0.01 # convert cm/s into m/s
		
	if  VERSION == 'New':
		CURRENT_X = np.array([[ 0.05,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.05],
		      [ 0.1 ,  0.4 ,  0.4 ,  0.4 ,  0.4 ,  0.1 ],
		      [ 0.1 ,  0.4 ,  0.5 ,  0.5 ,  0.4 ,  0.1 ],
		      [ 0.1 ,  0.4 ,  1.0 ,  0.8 ,  0.4 ,  0.1 ],
		      [ 0.1 ,  0.2 ,  0.2 ,  0.2 ,  0.2 ,  0.1 ],
		      [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ]])
		CURRENT_X = CURRENT_X*(FLOW_SPEED*0.01) # convert cm/s into m/s
	
    if FLOW_ON is 0:
        CURRENTVEL = ([0,0,0])
    if FLOW_ON is 1:
        CURRENTVEL = ([CURRENT_X, 0, 0])
    dCURRENTVEL = CURRENTVEL
    
    ## state Parameter
    ## set power for each state

    STATE_POWER = np.array([1e-12, 2e-4])

    ## define transition probability matrix for transitions among states
    TRANSMAT = np.array([[0.991, 0.009], [0.05, 0.95]])

    #### Preallocate
    VEL = np.zeros((round(SIMLENGTH/BINSIZE),NDIM))
    POS = np.zeros((round(SIMLENGTH/BINSIZE),NDIM))
    ORIENT = np.zeros((round(SIMLENGTH/BINSIZE,1)))
    POS[0,:] = IPOS
    ###### Setup Counters and state record
    PERIOD = 0
    TURNCOUNT = 0
    BINCOUNT = 0
    state = 0
    drag = 0.00075
    ###############################
    #### LAT LINE MODEL PARAMS ####
    ###############################

    SWIM_LEN = 0 # starting value
    movecount = 0 # starting value
    
    TAUlat = 0.85 ## increasing slows spiking
    latDOT = 0.0 ## starting value
    latRESET = 0
    latCUT = 1
    k_ = 0.007 ## decreasing causes increased spiking
    

    ###################
    #### MAIN LOOP ####
    ###################
    for ktime in range(0,int(SIMLENGTH/BINSIZE)-1):#(BINSIZE,SIMLENGTH,BINSIZE):

        TURNCOUNT += 1 ## Count Bins since last turn
        BINCOUNT += 1   ## COUNT BINS (absolute)
     
     ##  transition among states
        state = randpick(TRANSMAT[state])
        POWER= STATE_POWER[state]    
    
    ## if it is time to select new orientation, generate turns
        if TURNCOUNT > PERIOD:
            ## Select Interval Before Next Turn (bins)
	    PERIOD=round(np.random.exponential(LAMBDA,1)/BINSIZE)
            ## Determine Theta and Phi Distributions
            ## Select change in anlges
	
            if TURNINGDIST is 'normal':
                dTHETA = np.random.normal(0,SDA_T,1) 
                dPHI = np.random.normal(0,SDA_P,1)
            elif TURNINGDIST is 'uniform':
                dTHETA = np.random.uniform(-AMT_T, AMT_T,1)
                dPHI = np.random.uniform(-AMT_P,AMT_P,1)
	    
	    ##Update Angles
	    THETA += dTHETA
            PHI += dPHI

	    TURNCOUNT = 0
    
    ##################
    #### LAT LINE ####
    ##################

	if FLOW_ON is 1:
	    y_dot = int(POS[BINCOUNT-1][1]*39.37)
            z_dot = int(POS[BINCOUNT-1][2]*39.37)
	    CURRENT = CURRENT_X[y_dot][z_dot]
	    dCURRENTVEL = ([CURRENT, 0,0])
	elif FLOW_ON is 0:
	    CURRENT = 0
    
        if LATLINE == 1:
        
	    latDOT += (CURRENT + k_*(latDOT))*BINSIZE/TAUlat

        if latDOT >= latCUT:
	    latDOT = latRESET
	    SWIM_LEN = 0.25 + np.random.random(1)*1.75
            movecount = 0
                   
    ##########################    
    ###### Update FORCE ######
    ##########################
	
    ##Calculate Thrust (dimensional)
        if movecount < SWIM_LEN/BINSIZE:
            THRUST = np.array([8e-5, np.sin(THETA)*np.sin(PHI)*POWER,
			   np.cos(PHI)*POWER])
            movecount +=1
        elif movecount >= SWIM_LEN/BINSIZE:
            THRUST = np.array([np.cos(THETA)*np.sin(PHI)*POWER,
			   np.sin(THETA)*np.sin(PHI)*POWER, np.cos(PHI)*POWER])
	    
	ORIENT[BINCOUNT] = orient_angle(np.cos(THETA)*np.sin(PHI),np.sin(THETA)*np.sin(PHI))
    ## Calculate Current
    
    ##Calculate Drag
    #S= np.linalg.norm(dCURRENTVEL - VEL[BINCOUNT-1,:]) ##find apparent curr. magnitude
	if state is 1:
		#Fdrag=((1/2.)*AREA*Cdrag*S**2) ## caluclate drag force magnitude
	    Fdrag= drag*(dCURRENTVEL - VEL[BINCOUNT-1,:])#/S
	    if any(np.isnan(Fdrag)):
		Fdrag=0
	elif state is 0:
	    Fdrag = drag*(-VEL[BINCOUNT-1,:])

    ## caluclate force
        F = (THRUST.flatten() + GRAV*MASS + Fdrag)

    ###### Update Velocity ######
	VEL[BINCOUNT,:] = VEL[BINCOUNT-1] + F*BINSIZE/MASS ## F*BINSIZE/MASS = acceleration
    
    ###### Update Position ######
        POS[BINCOUNT,:] = POS[BINCOUNT-1] + (VEL[BINCOUNT,:]*BINSIZE)

    
    ###### If bounded mode is on, then confine position and velocity to bounds
        if BOUNDTEST is 1:
            for kdim in range(0,NDIM):
                if (POS[BINCOUNT,kdim]<MINS[kdim] or POS[BINCOUNT,kdim]>MAXS[kdim]):
                    VEL[BINCOUNT,kdim]=0
		
            POS[BINCOUNT,:]=np.maximum(MINS,POS[BINCOUNT,:])
            POS[BINCOUNT,:]=np.minimum(MAXS,POS[BINCOUNT,:])

    return POS,ORIENT
