import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
import matplotlib.patches as patches

import walkfunction as wlk
import plot_functions

LATLINE = True
FLOW_DYNAMICS = 'JEB'
data_nf = wlk.randwalk(300,0.1, 0, np.array([0.37, 0.075, 0.15]),latline = LATLINE, flow_version = FLOW_DYNAMICS)
data_f = wlk.randwalk(300,0.1, 0, data_nf['track'][-1],latline = LATLINE, flow_version = FLOW_DYNAMICS)

frogtrack_1 = np.zeros((len(data_nf['track'])*2,3))
frogtrack_1[0:len(data_nf['track']),:] = data_nf['track']
frogtrack_1[len(data_nf['track'])-1:-1,:] = data_f['track']
frogtrack_1 = frogtrack_1*100

LATLINE = True
data_nf = wlk.randwalk(300,0.1, 0, np.array([0.37, 0.075, 0.15]),latline = LATLINE, flow_version = FLOW_DYNAMICS)
data_f = wlk.randwalk(300,0.1, 0, data_nf['track'][-1],latline = LATLINE, flow_version = FLOW_DYNAMICS)
frogtrack_2 = np.zeros((len(data_nf['track'])*2,3))
frogtrack_2[0:len(data_nf['track']), :] = data_nf['track']
frogtrack_2[len(data_nf['track'])-1:-1, :] = data_f['track']
frogtrack_2 = frogtrack_2*100

flow = np.ones(len(frogtrack_2),'int')
flow[0:len(data_nf['track'])] = flow[0:len(data_nf['track'])]*0
speed = np.array(['0 cm/s','10 cm/s'],dtype='str')

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure(figsize=(10,8))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        ax1 = fig.add_subplot(2, 1, 1,projection='3d')
        ax2 = fig.add_subplot(2, 1, 2,projection='3d')
        
        self.t = np.arange(0,6000)
        self.x = frogtrack_1[self.t,0]
        self.y = frogtrack_1[self.t,1]
        self.z = frogtrack_1[self.t,2]
        
        self.x2 = frogtrack_2[self.t,0]
        self.y2 = frogtrack_2[self.t,1]
        self.z2 = frogtrack_2[self.t,2]
        #self.flow = flow[self.t]
        
        ax1.pbaspect = [1.4, 0.4, 0.4]
        ax2.pbaspect = [1.4, 0.4, 0.4]

        ax1.text2D(0.04, 0.65, '0 cm/s', fontsize=18, transform=ax1.transAxes)
        ax2.text2D(0.04, 0.65, '0 cm/s', fontsize=18, transform=ax2.transAxes)

        plot_functions.plot_tank(ax1)
        plot_functions.plot_tank(ax2)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        
        ax1.view_init(18, -95)
        ax2.view_init(18, -95)

        self.line1 = Line3D([], [], [], color='black')
        self.line1a = Line3D([], [], [], color='red', linewidth=2)
        self.line1e = Line3D([], [], [], color='red', marker='o', markeredgecolor='r')

        self.line2 = Line3D([], [], [], color='black')
        self.line2a = Line3D([], [], [], color='red', linewidth=2)
        self.line2e = Line3D([], [], [], color='red', marker='o', markeredgecolor='r')

        
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        
        ax2.add_line(self.line2)
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        
        ax1.set_xlim(0, 68)
        ax1.set_ylim(0, 15)
        ax1.set_zlim(0, 15)
        ax1.set_title('NO Lateral Line Model')
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
       
        plt.show()

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=False)

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
#ani.save('walk.mp4')
plt.show()
