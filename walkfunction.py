import numpy as np
import scipy as sp
from scipy.stats import rv_discrete

###################
# Model Parameteres
###################

# set number of dimensions in the simulation
ndim = 3

# define animal's size physical parameteres
bdrag = 0.00075 # this is presently being used in lew of the cdrad and area
area = 2e-4    # set the square area
cdrag = 0.0287 # according to Liu et al 1997
mass = 0.0004 

# Behavioral state parameters
nstates = 3
state_data = [{}]*nstates
state_data[0] = {'power': 0,                    # states 0 and 1 are the normal behavior
                 'orientation_type': 'animal',
                 'center_theta': None,
                 'center_phi': None,
                 'turn_lambda': 3,
                 'sda_t': 0.1*np.pi,
                 'sda_p': 0.1*np.pi,
                 }
state_data[1] = {'power': 8e-5,
                 'orientation_type': 'animal',
                 'center_theta': None,
                 'center_phi': None,
                 'turn_lambda': 3,
                 'sda_t': 0.1*np.pi,
                 'sda_p': 0.1*np.pi,
                 }
state_data[2] = {'power': 10e-5,                 # state 2 is triggered by the ll event (no input probability from 0,1)
                 'orientation_type': 'absolute',
                 'center_theta': np.pi,
                 'center_phi': -np.pi / 2,
                 'turn_lambda': 3,
                 'sda_t': 0.01*np.pi,
                 'sda_p': 0.01*np.pi,
                 }

transition_matrix = np.array([[0.991, 0.009, 0.0],  # define transition probability matrix for transitions among states (p = p / second)
                              [0.025, 0.975, 0.0],
                              [0.009, 0.0,   0.991]])



# Tank Boundaries
bounded = True        ## Turn boundries on or off (True or False)
min_bounds = np.array([0, 0, 0])       ## minimum boundries
max_bounds = np.array([0.685, 0.15, 0.15])## maximum boundries

# Set Gravity (deps on bouyancy)
gravity = np.array([0, 0, -.05])

# set lateral line model params
lat_tau = 0.135 # increasing slows spiking (time constant of neuron)
lat_reset = 0 # reset potential
lat_thresh = 1 # spike threshold
sensitivity = 1.25 # decreasing causes increased spiking
leak = 0.0001
###################
# Model Subfunctions
###################
current_data = {}
current_data['y'] = np.arange(0, 0.15, 0.15 / 6)
current_data['z'] = np.arange(0, 0.15, 0.15 / 6)
current_data['new'] = np.array([[ 0.08,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.15],
                              [ 0.15 ,  0.3 ,  0.3 ,  0.5 ,  0.3 ,  0.15 ],
                              [ 0.15 ,  0.5 ,  0.4 ,  0.4 ,  0.5 ,  0.15 ],
                              [ 0.15 ,  1.0 ,  0.6 ,  0.6 ,  1.0 ,  0.15 ],
                              [ 0.15 ,  0.5 ,  1.0 ,  1.0 ,  0.5 ,  0.15 ],
                              [ 0.10 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.10 ]])

current_data['jeb'] = np.array([[ 0.05,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.05],
                              [ 0.1 ,  0.4 ,  0.4 ,  0.4 ,  0.4 ,  0.1 ],
                              [ 0.1 ,  0.4 ,  0.5 ,  0.5 ,  0.4 ,  0.1 ],
                              [ 0.1 ,  0.4 ,  1.0 ,  0.8 ,  0.4 ,  0.1 ],
                              [ 0.1 ,  0.2 ,  0.2 ,  0.2 ,  0.2 ,  0.1 ],
                              [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ]])

def return_local_current(position, flow_version = 'new'):
    if flow_version.lower() in current_data:
        idx_y = np.argmin(abs(current_data['y'] - position[1]))
        idx_z = np.argmin(abs(current_data['z'] - position[2]))
        local_current_x= current_data[flow_version.lower()][idx_y, idx_z]
    else:
        raise Exception('No current data for flow field ' + flow_version)
    local_current_vector = np.array([local_current_x, 0, 0]) # presently there is no simulated current in x and y
    return local_current_vector

def randpick(TRANSMAT, current_state):
    p = rv_discrete(name = 'adhoc', values = (range(nstates),TRANSMAT[current_state,:]))
    new_state = p.rvs(size = 1)

    return new_state

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

def calculate_drag(s, form = 'linear'):
    if form.lower() == 'linear':
        d = s * bdrag
    elif form == 'quadratic':
        d = (0.5) * area * cdrag * s**2 # caclulate drag force magnitude
    return d

###################
# Model Main Function
###################
def randwalk(simlength, binsize, flow_speed, input_data, latline = True, flow_version = "JEB"):

    # Preallocate recording variables 
    if isinstance(input_data, np.ndarray):     #regular case of initial position
        time = np.arange(0,round(simlength/binsize)) * binsize
        velocity = np.zeros((round(simlength/binsize),ndim))
        position = np.zeros((round(simlength/binsize),ndim))
        orientation = np.zeros((round(simlength/binsize,1)))
        flow_rate = np.zeros((round(simlength/binsize,1)))
        state_record = np.zeros((round(simlength/binsize,1)))
        Vll = np.zeros((round(simlength/binsize,1)))
        position[0,:] = input_data
        bincount = 0
    elif isinstance(input_data, dict):       # continued run on data
        time = np.append(input_data['time'], input_data['time'][-1] + np.arange(0,round(simlength/binsize)) * binsize)
        velocity = np.append(input_data['velocity'], np.zeros((round(simlength/binsize),ndim)), axis = 0)
        position = np.append(input_data['track'], np.zeros((round(simlength/binsize),ndim)), axis = 0)
        orientation = np.append(input_data['orientation'], np.zeros((round(simlength/binsize,1))), axis = 0)
        flow_rate = np.append(input_data['flow_rate'], np.zeros((round(simlength/binsize,1))), axis = 0)
        state_record = np.append(input_data['state'], np.zeros((round(simlength/binsize,1))), axis = 0)
        Vll = np.append(input_data['lateral_line'], np.zeros((round(simlength/binsize,1))), axis = 0)
        bincount = input_data['time'].shape[0] - 1

    # normalize the transition matrix with the binsize
    TRANSMAT = transition_matrix #* binsize

    # Set Flow
    if flow_speed == 0:
        flow_on = False
    elif flow_speed > 0:
        flow_on = True

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

    # setup counters
    turncount = 0

    # MAIN LOOP 
    for ktime in range(0,int(simlength/binsize)-1): #(binsize,simlength,binsize):
        bincount += 1   # COUNT BINS (absolute)
        turncount += 1  # Count Bins since last turn
             
        # transition among states
        state = randpick(TRANSMAT, state)

        power = state_data[state]['power'] # set the power    
    
        # if it is time to select new orientation, generate turns
        if turncount > period:
            turncount = 0 # reset the turn counter

            period=round(np.random.exponential(state_data[state]['turn_lambda'], 1) / binsize) # Select Interval Before Next Turn (bins)
            
            # setup turning distribution
            if state_data[state]['orientation_type'] == "animal":
                center_theta = theta
                center_phi = phi
            elif state_data[state]['orientation_type'] == "absolute":
                center_theta = state_data[state]['center_theta']
                center_phi = state_data[state]['center_phi']

            # perform turns
            theta = center_theta + np.random.normal(0,state_data[state]['sda_t'],1) # add change in anlges from normal distribution
            phi = center_phi + np.random.normal(0,state_data[state]['sda_p'],1)

        # find current at the present position
        if flow_on:
            local_current = flow_speed * (.01) * return_local_current(position[bincount-1, :], flow_version = flow_version) # return the local current in cm/s
        else:
            local_current = [0, 0, 0]
        apparent_current = local_current - velocity[bincount-1,:]
              
        # calculate thrust
        Fthrust = np.array([np.cos(theta)*np.sin(phi)*power,
                           np.sin(theta)*np.sin(phi)*power, 
                           np.cos(phi)*power])
    
        # calculate drag
        Fdrag = bdrag * apparent_current
        
        # sum forces
        F = (Fthrust.flatten() + Fdrag.flatten() + gravity * mass)
        # integrate    
        velocity[bincount,:] = velocity[bincount-1] + F*binsize/mass # Update Velocity 
        position[bincount,:] = position[bincount-1] + (velocity[bincount,:]*binsize) # Update Position

        # If bounded mode is on, then confine position and velocity to bounds
        if bounded:
            for kdim in range(0, ndim):
                if (position[bincount,kdim]<min_bounds[kdim] or position[bincount,kdim]>max_bounds[kdim]):
                    velocity[bincount,kdim]=0
            position[bincount,:]=np.maximum(min_bounds, position[bincount,:])
            position[bincount,:]=np.minimum(max_bounds, position[bincount,:])

        # run lateral line model
        if latline:
            lat_v += (sensitivity * local_current[0] + leak * (0 - lat_v))*binsize/lat_tau
            Vll[bincount] = lat_v
        if lat_v >= lat_thresh:
            period = 0
            state = 2 # set the state to the trigger
            lat_v = lat_reset


        # record data
        flow_rate[bincount] = local_current[0]
        state_record[bincount] = state
        orientation[bincount] = orient_angle(np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi)) # record orientation angle

    # pack output data
    data = {'time': time,
            'track': position,
            'velocity': velocity,
            'orientation': orientation,
            'lateral_line': Vll,
            'flow_rate': flow_rate,
            'state': state_record}

    return data


# script to execute example
if __name__ == "__main__":
    latline = True
    flow = 'JEB'
    data = randwalk(300, 0.1, 0, np.array([0.37, 0.075, 0.15]), latline = latline, flow_version = flow)
    data = randwalk(300, 0.1, 2, data, latline = latline, flow_version = flow)
    import plot_functions
    axes = plot_functions.plot_tank()
    plot_functions.plot_track3d(data['track'], axes = axes)
    plot_functions.plot_xdata(data)
    plot_functions.plot_kineticdata(data)
    plot_functions.plt.show()




