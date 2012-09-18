import numpy as np
import scipy as sp

###################
# Model Parameteres
###################

# set number of dimensions in the simulation
ndim = 3

# define animal's size physical parameteres
drag = 0.00075 # this is presently being used in lew of the cdrad and area
area = 2e-4    # set the square area
cdrag = 0.0287 # according to Liu et al 1997
mass = 0.0004 

# Behavioral state parameters
state_power = np.array([1e-12, 2e-4]) # set power for each state
TRANSMAT = np.array([[0.991, 0.009],  # define transition probability matrix for transitions among states
                     [0.05, 0.95]])

# Turning Parameters
turn_lambda = 1 # average length of time between turns
sda_t = 0.1*np.pi # standard deviation of turning angles for normal turning dist
sda_p = 0.1*np.pi

# Tank Boundaries
bounded = True        ## Turn boundries on or off (True or False)
min_bounds = np.array([0, 0, 0])       ## minimum boundries
max_bounds = np.array([0.685, 0.15, 0.15])## maximum boundries

# Set Gravity (deps on bouyancy)
gravity = np.array([0, 0, -.05])

# set lateral line model params
lat_tau = 0.1 # increasing slows spiking (time constant of neuron)
lat_reset = 0 # reset potential
lat_thresh = 1 # spike threshold
k_ = 0.006 # decreasing causes increased spiking

###################
# Model Subfunctions
###################
def generate_current_plane(flow_version):
    if flow_version.lower() == 'new':
        current_plane = np.array([[ 0.08,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.15],
              [ 0.15 ,  0.3 ,  0.3 ,  0.5 ,  0.3 ,  0.15 ],
              [ 0.15 ,  0.5 ,  0.4 ,  0.4 ,  0.5 ,  0.15 ],
              [ 0.15 ,  1.0 ,  0.6 ,  0.6 ,  1.0 ,  0.15 ],
              [ 0.15 ,  0.5 ,  1.0 ,  1.0 ,  0.5 ,  0.15 ],
              [ 0.10 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.10 ]])
    elif flow_version.lower() == 'jeb':
        current_plane = np.array([[ 0.05,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.05],
              [ 0.1 ,  0.4 ,  0.4 ,  0.4 ,  0.4 ,  0.1 ],
              [ 0.1 ,  0.4 ,  0.5 ,  0.5 ,  0.4 ,  0.1 ],
              [ 0.1 ,  0.4 ,  1.0 ,  0.8 ,  0.4 ,  0.1 ],
              [ 0.1 ,  0.2 ,  0.2 ,  0.2 ,  0.2 ,  0.1 ],
              [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ]])
    return current_plane

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

###################
# Model Main Function
###################
def randwalk(simlength, binsize, flow_speed, Iposition, latline = True, flow_version = "JEB"):

    # Set Flow
    if flow_speed == 0:
        flow_on = False
    elif flow_speed > 0:
        flow_on = True
        current_plane = generate_current_plane(flow_version)
        current_plane = current_plane * flow_speed*0.01 # convert cm/s into m/s

    # set initial orientation angles
    theta = 0 
    phi = 0

    # setup state records
    state = 0
    period = 0

    # setup lateral line state records
    SWIM_LEN = 0 # starting value
    movecount = 0 # starting value
    lat_v = 0.0 # starting value (resting potential)

    # Preallocate recording variables
    time = np.arange(0,round(simlength/binsize)) * binsize
    velocity = np.zeros((round(simlength/binsize),ndim))
    position = np.zeros((round(simlength/binsize),ndim))
    orientation = np.zeros((round(simlength/binsize,1)))
    flow_rate = np.zeros((round(simlength/binsize,1)))
    Vll = np.zeros((round(simlength/binsize,1)))
    position[0,:] = Iposition
    
    # setup counters
    turncount = 0
    bincount = 0

    # MAIN LOOP 
    for ktime in range(0,int(simlength/binsize)-1): #(binsize,simlength,binsize):
        bincount += 1   ## COUNT BINS (absolute)
        turncount += 1 ## Count Bins since last turn
             
        # transition among states
        state = randpick(TRANSMAT[state])
        power = state_power[state]    
    
        # if it is time to select new orientation, generate turns
        if turncount > period:
            # Select Interval Before Next Turn (bins)
            period=round(np.random.exponential(turn_lambda, 1) / binsize)
            turncount = 0
            # add change in anlges from normal distribution
            theta += np.random.normal(0,sda_t,1) 
            phi += np.random.normal(0,sda_p,1)

        # find current at the present position
        if flow_on:
            y_dot = int(position[bincount-1][1]*39.37)
            z_dot = int(position[bincount-1][2]*39.37)
            local_current_x = current_plane[y_dot][z_dot]
            local_current_vel = [local_current_x, 0,0]
        else:
            local_current_x = 0
            local_current_vel = [0, 0, 0]
        flow_rate[bincount] = local_current_x

        # lateral line model
        if latline:
            lat_v += (local_current_x + k_*(lat_v))*binsize/lat_tau
            Vll[bincount] = lat_v
        if lat_v >= lat_thresh:
            lat_v = lat_reset
            SWIM_LEN = 0.25 + np.random.random(1)*1.75
            movecount = 0
                    
        # Calculate Thrust (dimensional)
        if movecount < SWIM_LEN/binsize: # if in swimming state, swimm towards flow
            thrust = np.array([8e-5, 
                            np.sin(theta)*np.sin(phi)*power,
                            np.cos(phi)*power])
            movecount += 1
        elif movecount >= SWIM_LEN/binsize: # otherwise swim normally
            thrust = np.array([np.cos(theta)*np.sin(phi)*power,
                               np.sin(theta)*np.sin(phi)*power, 
                               np.cos(phi)*power])
    
        #Calculate Drag
        #S= np.linalg.norm(dCURRENTvelocity - velocity[bincount-1,:]) ##find apparent curr. magnitude
        if state is 1:
            #Fdrag=((1/2.)*area*cdrag*S**2) ## caluclate drag force magnitude
            Fdrag= drag*(local_current_vel - velocity[bincount-1,:])#/S
            if any(np.isnan(Fdrag)):
                Fdrag=0
        elif state is 0:
            Fdrag = drag*(-velocity[bincount-1,:])

        # caluclate force
        F = (thrust.flatten() + gravity * mass + Fdrag)
        # integrate    
        velocity[bincount,:] = velocity[bincount-1] + F*binsize/mass # Update Velocity 
        position[bincount,:] = position[bincount-1] + (velocity[bincount,:]*binsize) # Update Position

        # record orientation angle
        orientation[bincount] = orient_angle(np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi))

        # If bounded mode is on, then confine position and velocity to bounds
        if bounded:
            for kdim in range(0, ndim):
                if (position[bincount,kdim]<min_bounds[kdim] or position[bincount,kdim]>max_bounds[kdim]):
                    velocity[bincount,kdim]=0
            position[bincount,:]=np.maximum(min_bounds, position[bincount,:])
            position[bincount,:]=np.minimum(max_bounds, position[bincount,:])

    # pack output data
    data = {'time': time,
            'track': position,
            'orientation': orientation,
            'lateral_line': Vll,
            'flow_rate': flow_rate}

    return data


# script to execute example
if __name__ == "__main__":
    latline = True
    flow = 'JEB'
    data = randwalk(300, 0.1, 5, np.array([0.37, 0.075, 0.15]), latline = latline, flow_version = flow)

    import plot_functions
    axes = plot_functions.plot_tank()
    plot_functions.plot_track3d(data['track'], axes = axes)
    plot_functions.plot_xdata(data)
    plot_functions.plt.show()




