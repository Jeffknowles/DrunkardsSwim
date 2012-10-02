import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
import walkfunction as wlk
import plot_functions

uname = 'jeffknowles'

FLOW_DYNAMICS = 'JEB'

# Decide whether to use real data
REAL_DATA = 1 # 1 = Yes

t_noflow = 300 # in sec
t_flow = 300 # in sec

binsize = 1.0/30.0 # in sec (1sec/30FPS if using real data)
DOWNSAMPLE_RATE = 4 

current = 8 # in cm/s.

if REAL_DATA == 1:
    # Import No Flow data:
    frog1_NF = np.genfromtxt('/Users/' + uname + '/Dropbox/TadpoleStuff/Tracked_data_CSV/NPF5_2_NF_lessFilt.csv',delimiter=',')
    
    # filter zeros downsample
    index = frog1_NF[:,0]!=0
    frog1_NF = frog1_NF[index]
    frog1_NF = frog1_NF[0::DOWNSAMPLE_RATE]

    # reset the simulation length to reflect data
    t_noflow = len(frog1_NF)*binsize*DOWNSAMPLE_RATE

    # Import Flow data:
    frog1_F = np.genfromtxt('/Users/' + uname + '/Dropbox/TadpoleStuff/Tracked_data_CSV/NPF5_2_F_lessFilt.csv',delimiter=',')

    # filter zeros and downsample
    index = frog1_F[:,0]!=0
    frog1_F = frog1_F[index]
    frog1_F = frog1_F[0::DOWNSAMPLE_RATE]

    # reset the simulation length to reflect data
    t_flow = len(frog1_F)*binsize*DOWNSAMPLE_RATE

    # Append data into one structure:
    frog1 = np.append(frog1_NF,frog1_F,axis=0)
    frog1 = {'track': frog1,
             'state': None}
    
    FROG1NAME = 'Observed Data'
else:   
    # generate all data w/ lateral line switched off
    LATLINE = False
    frog1 = wlk.randwalk(t_noflow, binsize, 0, np.array([0.37, 0.075, 0.15]), latline = LATLINE, flow_version = FLOW_DYNAMICS)
    frog1 = wlk.randwalk(t_flow, binsize, current, frog1,latline = LATLINE, flow_version = FLOW_DYNAMICS)
    frog1['track'] = frog1['track']*100
    FROG1NAME = 'NO Lateral Line Model'
    # downsample
    for data in frog1:
        frog1[data] = frog1[data][0::DOWNSAMPLE_RATE]
    
# generate all data w/ lateral line switched on - First compute at high resolution before downsampling
LATLINE = True
frog2 = wlk.randwalk(t_noflow, binsize, 0, np.array([0.34, 0.075, 0.15]), latline = LATLINE, flow_version = FLOW_DYNAMICS)
frog2 = wlk.randwalk(t_flow, binsize, current, frog2,latline = LATLINE, flow_version = FLOW_DYNAMICS)
frog2['track'] = frog2['track']*100

state_colors = ['blue', 'black', 'red']

# downsample
for data in frog2:
    frog2[data] = frog2[data][0::DOWNSAMPLE_RATE]

flow = np.ones(frog1['track'].shape[0],'int') * current
flow[0: (float(t_noflow)/ binsize / DOWNSAMPLE_RATE)] = 0

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure(facecolor = 'w', figsize = [12, 6])
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        ax1 = fig.add_subplot(2, 1, 1,projection='3d')
        ax2 = fig.add_subplot(2, 1, 2,projection='3d')
        
        self.t = np.arange(0,len(frog1['track']))
        self.x = frog1['track'][self.t,0]
        self.y = frog1['track'][self.t,1]
        self.z = frog1['track'][self.t,2]
        
        self.x2 = frog2['track'][self.t,0]
        self.y2 = frog2['track'][self.t,1]
        self.z2 = frog2['track'][self.t,2]
        self.flow = flow[self.t]
        self.state1 = frog1['state']
        self.state2 = frog2['state']

        
        self.flowtexta1 = ax1.text2D(0.001, 0.65, 'Flow 0 cm/s', fontsize=18, transform=ax1.transAxes)
        self.flowtexta2 = ax2.text2D(0.001, 0.65, 'Flow 0 cm/s', fontsize=18, transform=ax2.transAxes)
        if REAL_DATA == 0:
            self.statetexta1 = ax1.text2D(0.001, 0.50, 'State 0', fontsize=18, transform=ax1.transAxes)
        self.statetexta2 = ax2.text2D(0.001, 0.50, 'State 0', fontsize=18, transform=ax2.transAxes)
        self.framerate = ax1.text2D(0.90, 0.85, str(DOWNSAMPLE_RATE) + 'x speed', fontsize = 18, transform=ax1.transAxes)
        self.info1 = ax2.text2D(0.90,0.95,'State key',fontsize = 18,transform=ax2.transAxes)
        self.info2 = ax2.text2D(0.90,0.90,'------------',fontsize = 18, transform=ax2.transAxes)
        self.info2 = ax2.text2D(0.90,0.75,'Blue: rest',fontsize = 18, transform=ax2.transAxes, color = 'blue')
        self.info3 = ax2.text2D(0.90,0.60,'Black: swim ', fontsize = 18, transform=ax2.transAxes, color = 'black')
        self.info4 = ax2.text2D(0.90,0.45,'Red: L.L. orientation.', fontsize = 18, transform=ax2.transAxes, color = 'red')


        plot_functions.plot_tank(ax1)
        plot_functions.plot_tank(ax2)

        #ax1.set_xlabel('x')
        #ax1.set_ylabel('y')
        #ax1.set_zlabel('z')

        #ax2.set_xlabel('x')
        #ax2.set_ylabel('y')
        #ax2.set_zlabel('z')
        
        self.line1 = Line3D([], [], [], color='black')
        self.line1a = Line3D([], [], [], color='green', linewidth=2)
        self.line1e = Line3D([], [], [], color='green', marker='o')

        self.line2 = Line3D([], [], [], color='black')
        self.line2a = Line3D([], [], [], color='red', linewidth=2)
        self.line2e = Line3D([], [], [], color='red', marker='o')

        
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        
        ax2.add_line(self.line2)
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        
        ax1.set_xlim(0, 68)
        ax1.set_ylim(0, 15)
        ax1.set_zlim(0, 15)
        ax1.set_title(FROG1NAME)
        ax1.set_aspect('equal')

        ax2.set_xlim(0, 68)
        ax2.set_ylim(0, 15)
        ax2.set_zlim(0, 15)
        ax2.set_title('Lateral Line Model')
        ax2.set_aspect('equal')
        
        ax1.set_xticks([0, 68])
        ax1.set_yticks([])
        ax1.set_zticks([0, 15])

        ax2.set_xticks([0, 68])
        ax2.set_yticks([])
        ax2.set_zticks([0, 15])

        plt.rc('xtick', labelsize=20)

        animation.TimedAnimation.__init__(self, fig, interval=10, blit=False)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_len = 10
        head_slice = (self.t > self.t[i] - 7.0) & (self.t < self.t[i])

        self.line1.set_data(self.x[:i], self.y[:i])
        self.line1a.set_data(self.x[head_slice], self.y[head_slice])
        self.line1e.set_data(self.x[head], self.y[head])
        self.line1.set_3d_properties(self.z[:i])
        self.line1a.set_3d_properties(self.z[head_slice])
        self.line1e.set_3d_properties(self.z[head])

        self.line2.set_data(self.x2[:i], self.y2[:i])
        self.line2a.set_data(self.x2[head_slice], self.y2[head_slice])
        self.line2e.set_data(self.x2[head], self.y2[head])
        self.line2.set_3d_properties(self.z2[:i])
        self.line2a.set_3d_properties(self.z2[head_slice])
        self.line2e.set_3d_properties(self.z2[head])
        self.line2a.set_color(color = state_colors[int(self.state2[head])])
        self.line2e.set_color(color = state_colors[int(self.state2[head])])

        self.flowtexta1.set_text('flow ' + str(self.flow[head]) + ' cm/s')
        self.flowtexta2.set_text('flow ' + str(self.flow[head]) + ' cm/s')
        if REAL_DATA == 0:
            self.statetexta1.set_text('state ' + str(self.state1[head]))
        self.statetexta2.set_text('state ' + str(int(self.state2[head])))
        self.statetexta2.set_color(state_colors[int(self.state2[head])])
        

        self._drawn_artists = [self.line1, self.line1a, self.line1e,
                               self.line2, self.line2a, self.line2e]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines =  [self.line1, self.line1a, self.line1e,
                  self.line2, self.line2a, self.line2e]
        for l in lines:
            l.set_data([], [])

ani = SubplotAnimation()
#ani.save('walk.mp4', fps=20, codec='mpeg4', clear_temp=True, frame_prefix='_tmp')
plt.show()

