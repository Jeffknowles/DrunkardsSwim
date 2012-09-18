# plot_functions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D

def plot_tank(axes = None):
    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection = '3d')
        axes.set_xlim(0, 68)
        axes.set_ylim(0, 15)
        axes.set_zlim(0, 15)

    c1 = '.3' 
    c2 = '.6'
    box = {}
    box['bottom1'] = Line3D([0, 68], [0, 0], [0,0], color = c1, linewidth = 2)
    box['bottom2'] = Line3D([0, 0], [0, 15], [0,0], color = c2, linewidth = 2)
    box['bottom3'] = Line3D([0, 68], [15, 15], [0,0], color = c2, linewidth = 2)
    box['bottom4'] = Line3D([68, 68], [0, 15], [0,0], color = c1, linewidth = 2)
    box['top1'] = Line3D([0, 68], [0, 0], [15, 15], color = c1, linewidth = 2)
    box['top2'] = Line3D([0, 0], [0, 15], [15, 15], color = c1, linewidth = 2)
    box['top3'] = Line3D([0, 68], [15, 15], [15, 15], color = c1, linewidth = 2)
    box['top4'] = Line3D([68, 68], [0, 15], [15, 15], color = c1, linewidth = 2)
    box['side1'] = Line3D([0, 0], [0, 0], [0, 15], color = c1, linewidth = 2)
    box['side2'] = Line3D([68, 68], [0, 0], [0, 15], color = c1, linewidth = 2)
    box['side3'] = Line3D([68, 68], [15, 15], [0, 15], color = c2, linewidth = 2)
    box['side4'] = Line3D([0, 0], [15, 15], [0, 15], color = c2, linewidth = 2)

    keylist = box.keys()
    keylist.sort()
    for key in keylist:
        axes.add_line(box[key])

    axes.view_init(25, -105)
    axes.set_axis_bgcolor('w')

    return axes

def plot_track3d(track, axes = None):
    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection = '3d')
        axes.set_xlim(0, 68)
        axes.set_ylim(0, 15)
        axes.set_zlim(0, 15)
    track = track * 100
    track_plot = Line3D(track[:,0], track[:, 1], track[:, 2], color = 'black', linewidth = 2)
    axes.add_line(track_plot)
    axes.pbaspect = [1, 5, 1]
    axes.set_aspect('equal')
    return axes
    
def plot_xdata(data):
    fig = plt.figure()

    a = {}
    if 'track' in data:
        a[1] = fig.add_subplot(411)    
        plt.plot(data['time'], data['track'][:,0] * 100, color = 'black')
        a[1].set_ylim(0, 68)
        a[1].set_xlabel('Time (s)')
        a[1].set_ylabel('X Position (cm)')

    if 'orientation' in data:
        a[2] = fig.add_subplot(412)    
        plt.plot(data['time'], data['orientation'], color = 'black')
        a[2].set_ylim(0, 360)
        a[2].set_xlabel('Time (s)')
        a[2].set_ylabel('Orientation angle')

    if 'flow_rate' in data:
        a[3] = fig.add_subplot(413)    
        plt.plot(data['time'], data['flow_rate']*100, color = 'black')
        a[3].set_xlabel('Time (s)')
        a[3].set_ylabel('Flow Rate (cm/s)')

    if 'lateral_line' in data:
        a[4] = fig.add_subplot(414)    
        plt.plot(data['time'], data['lateral_line'], color = 'black')
        a[4].set_ylim(0, 1)
        a[4].set_xlabel('Time (s)')
        a[4].set_ylabel('LL Potential')

    return a





    


if __name__ == "__main__":
    plot_tank()
