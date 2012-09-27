import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import walkfunction as wlk
import plot_functions

FLOW_DYNAMICS = 'JEB'

# Decide whether to use real data
FROG1NAME = 'lateral line model'
REAL_DATA = 0

t_noflow = 10 # in sec
t_flow = 60 # in sec

binsize = 1.0/30.0 # in sec (1sec/30FPS if using real data)
DOWNSAMPLE_RATE = 4

current = 8 # in cm/s.

LATLINE = True
frog1 = wlk.randwalk(t_noflow, binsize, 0, np.array([0.34, 0.075, 0.15]), latline = LATLINE,
                     flow_version = FLOW_DYNAMICS)
frog1 = wlk.randwalk(t_flow, binsize, current, frog1,latline = LATLINE, flow_version = FLOW_DYNAMICS)
frog1['track'] = frog1['track']*100
frog1['flow_rate'] = frog1['flow_rate'] * 100
# downsample all fields:
for data in frog1:
    frog1[data] = frog1[data][0::DOWNSAMPLE_RATE]



flow = np.ones(frog1['track'].shape[0],'int') * current
flow[0: (float(t_noflow)/ binsize / DOWNSAMPLE_RATE)] = 0

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure(facecolor = 'w', figsize = [12, 6])
        #fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        ax1 = plt.subplot2grid((3,3), (0,0), colspan=3,projection='3d')
        ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
        ax3 = plt.subplot2grid((3,3), (2, 0), colspan=2)
        ax4 = plt.subplot2grid((3,3), (1, 2), rowspan=2)


        
        self.t = np.arange(0,len(frog1['track']))
        self.x = frog1['track'][self.t,0]
        self.y = frog1['track'][self.t,1]
        self.z = frog1['track'][self.t,2]
        
        self.x2 = frog1['time'][self.t]
        self.y2 = frog1['flow_rate'][self.t]
        self.y3 = frog1['lateral_line'][self.t]
       

        ax2.text(0.01, 0.8, str(current) + ' cm/s', fontsize=16, transform=ax2.transAxes)
        ax2.text(0.01, 0.05, '0 cm/s', fontsize=16, transform=ax2.transAxes)
        ax3.text(0.01, 0.8, 'threshold',fontsize=16, transform=ax3.transAxes)
        ax3.text(0.01, 0.08, 'baseline', fontsize=16, transform=ax3.transAxes)


        plot_functions.plot_tank(ax1)
        plot_functions.plot_flow(current,ax2)
        plot_functions.plot_leak(ax3)
        plot_functions.plot_flow_field('jeb',axes = ax4)

        
        self.line1 = Line3D([], [], [], color='black')
        self.line1a = Line3D([], [], [], color='red', linewidth=2)
        self.line1e = Line3D([], [], [], color='red', marker='o', markeredgecolor='r')

        self.line2 = Line2D([], [], color='black')
        self.line2e = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        
        self.line3 = Line2D([], [], color='red', linewidth=2)
        self.line3e = Line2D([], [], color='red', marker='o', markeredgecolor='r')

        self.line4 = Line2D([],[], color='red', marker='o', markeredgecolor='r', markersize=10)
        
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        
        ax2.add_line(self.line2)
        ax2.add_line(self.line2e)

        ax3.add_line(self.line3)
        ax3.add_line(self.line3e)

        ax4.add_line(self.line4)
        #ax4.add_line(self.line4e)
        
        ax1.set_xlim(0, 68)
        ax1.set_ylim(0, 15)
        ax1.set_zlim(0, 15)
        ax1.set_title(FROG1NAME)
        ax1.set_aspect('equal')

        ax2.set_xlim(0, len(frog1['time'])*binsize*DOWNSAMPLE_RATE)
        ax2.set_ylim(-0.05, 1.05 * current)
        ax2.axis('off')

        ax3.set_xlim(0, len(frog1['time'])*binsize*DOWNSAMPLE_RATE)
        ax3.set_ylim(-0.05, 1.05)
        ax3.axis('off')

        ax4.set_title('current field')
        ax4.set_aspect('equal')
        
        ax1.set_xticks([0, 68])
        ax1.set_yticks([])
        ax1.set_zticks([0, 15])

        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3.set_xticks([])
        ax3.set_yticks([])
        
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        plt.rc('xtick', labelsize=16)
        #plt.rc('title', labelsize=18)
        plt.tight_layout()
        
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
        self.line2e.set_data(self.x2[head], self.y2[head])
        
        self.line3.set_data(self.x2[:i], self.y3[:i])
        self.line3e.set_data(self.x2[head], self.y3[head])
        
        self.line4.set_data(self.y[i]/15.0*5.0, self.z[i]/15.0*5.0)
        #self.line4.set_data(self.y[he
        

        self._drawn_artists = [self.line1, self.line1a, self.line1e,
                               self.line2,self.line2e, self.line3,
                               self.line3e,self.line4]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines =  [self.line1, self.line1a, self.line1e,
                  self.line2,self.line2e,self.line3,
                  self.line3e, self.line4]
        for l in lines:
            l.set_data([], [])

ani = SubplotAnimation()
ani.save('walk.mp4', fps=20, codec='mpeg4', clear_temp=True, frame_prefix='_tmp')
plt.show()
