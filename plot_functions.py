# plot_functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.lines import Line2D 

def plot_tank(axes = None):
    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection = '3d')
        axes.set_xlim(0, 68.5)
        axes.set_ylim(0, 15)
        axes.set_zlim(0, 15)

    c1 = '.3' 
    c2 = '.6'
    box = {}
    box['bottom1'] = Line3D([0, 68.5], [0, 0], [0,0], color = c1, linewidth = 2)
    box['bottom2'] = Line3D([0, 0], [0, 15], [0,0], color = c2, linewidth = 2)
    box['bottom3'] = Line3D([0, 68.5], [15, 15], [0,0], color = c2, linewidth = 2)
    box['bottom4'] = Line3D([68.5, 68.5], [0, 15], [0,0], color = c1, linewidth = 2)
    box['side1'] = Line3D([0, 0], [0, 0], [0, 15], color = c1, linewidth = 2)
    box['side2'] = Line3D([68.5, 68.5], [0, 0], [0, 15], color = c1, linewidth = 2)
    box['side3'] = Line3D([68.5, 68.5], [15, 15], [0, 15], color = c2, linewidth = 2)
    box['side4'] = Line3D([0, 0], [15, 15], [0, 15], color = c2, linewidth = 2)
    box['top1'] = Line3D([0, 68.5], [0, 0], [15, 15], color = c1, linewidth = 2)
    box['top2'] = Line3D([0, 0], [0, 15], [15, 15], color = c1, linewidth = 2)
    box['top3'] = Line3D([0, 68.5], [15, 15], [15, 15], color = c1, linewidth = 2)
    box['top4'] = Line3D([68.5, 68.5], [0, 15], [15, 15], color = c1, linewidth = 2)

    keylist = box.keys()
    keylist.sort()
    for key in keylist:
        axes.add_line(box[key])

    axes.view_init(25, -100)
    axes.set_axis_bgcolor('w')

    return axes

def plot_track3d(track, axes = None):
    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection = '3d')
        axes.set_xlim(0, 68.5)
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
        a[1].set_ylim(0, 68.5)
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
        plt.plot(data['time'], data['flow_rate'] * 100, color = 'black')
        a[3].set_xlabel('Time (s)')
        a[3].set_ylabel('Flow Rate (cm/s)')

    if 'lateral_line' in data:
        a[4] = fig.add_subplot(414)    
        plt.plot(data['time'], data['lateral_line'], color = 'black')
        a[4].set_ylim(0, 1)
        a[4].set_xlabel('Time (s)')
        a[4].set_ylabel('LL Potential')

    return a



def plot_kineticdata(data):
    fig = plt.figure()
    a = {}

    if 'state' in data:
        a[0] = fig.add_subplot(411)    
        plt.plot(data['time'], data['state'], color = 'black')
        a[0].set_ylim(-.1, 2.1)
        a[0].set_xlabel('Time (s)')
        a[0].set_ylabel('State')
    if 'velocity' in data:
        a[1] = fig.add_subplot(412)    
        speed = np.array([np.linalg.norm(vel) for vel in data['velocity']])
        plt.plot(data['time'], speed * 100, color = 'black')
        #a[1].set_ylim(0, 68.5)
        a[1].set_xlabel('Time (s)')
        a[1].set_ylabel('Speed (cm/s)')

    if 'lateral_line' in data:
        a[4] = fig.add_subplot(413)    
        plt.plot(data['time'], data['lateral_line'], color = 'black')
        a[4].set_ylim(0, 1)
        a[4].set_xlabel('Time (s)')
        a[4].set_ylabel('LL Potential')

    return fig

def plot_leak(axes = None):
    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection = '3d')
        axes.set_xlim(-0.05, 1.05)
        #axes.set_ylim(0, 15)


    c1 = '.6' 
    box = {}
    box['bottom1'] = Line2D([0, 100], [0, 0], color = c1, linewidth = 2)
    box['top1'] = Line2D([0, 100], [1, 1], color = c1, linewidth = 2)


    keylist = box.keys()
    keylist.sort()
    for key in keylist:
        axes.add_line(box[key])

    axes.set_axis_bgcolor('w')
    
    return axes

def plot_flow(current,axes = None):
    if axes == None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.set_xlim(-0.05, 1.05)
        #axes.set_ylim(0, 15)

    c1 = '.6' 
    box = {}
    box['bottom1'] = Line2D([0, 100], [0, 0], color = c1, linewidth = 2)
    box['top1'] = Line2D([0, 100], [current,current], color = c1, linewidth = 2)


    keylist = box.keys()
    keylist.sort()
    for key in keylist:
        axes.add_line(box[key])

    axes.set_axis_bgcolor('w')
    
    return axes

def plot_leak2D(data, axes = None):
    if axes == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if 'lateral_line' in data:

        ax.plot(data['lateral_line'], 'k')
        ax.plot(np.ones(data['time'].shape[0]),'--k',linewidth=1)
        ax.plot(np.zeros(data['time'].shape[0]),'--k',linewidth=1)
        ax.text(0.01, 0.99, 'threshold', fontsize=20,transform=ax.transAxes)
        ax.text(0.01, 0.01, 'baseline',  fontsize=20,transform=ax.transAxes)
        plt.ylim(-0.05,1.05)
        plt.axis('off')
    return fig

def plot_current2D(data, current, axes = None):
    if axes == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if 'flow_rate' in data:

        ax.plot(data['flow_rate'], 'k')
        ax.plot(np.ones(data['time'].shape[0]) * 100 * current,'--k',linewidth=1)
        ax.plot(np.zeros(data['time'].shape[0]),'--k',linewidth=1)
        ax.text(0.01, 0.99, str(current) + 'cm/s', fontsize=20,transform=ax.transAxes)
        ax.text(0.01, 0.01, '0 cm/s',  fontsize=20,transform=ax.transAxes)
        plt.ylim(-0.1,1.025*current)
        plt.axis('off')
    return fig
    
def plot_flow_field3D(flow_field = 'jeb',axes = None):
    
    if flow_field.lower() == 'new':
        current_data = np.array([[ 0.08,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.15],
                              [ 0.15 ,  0.3 ,  0.3 ,  0.5 ,  0.3 ,  0.15 ],
                              [ 0.15 ,  0.5 ,  0.4 ,  0.4 ,  0.5 ,  0.15 ],
                              [ 0.15 ,  1.0 ,  0.6 ,  0.6 ,  1.0 ,  0.15 ],
                              [ 0.15 ,  0.5 ,  1.0 ,  1.0 ,  0.5 ,  0.15 ],
                              [ 0.10 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.10 ]])
    if flow_field.lower() == 'jeb':
        current_data = np.array([[ 0.05,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.05],
                              [ 0.1 ,  0.4 ,  0.4 ,  0.4 ,  0.4 ,  0.1 ],
                              [ 0.1 ,  0.4 ,  0.5 ,  0.5 ,  0.4 ,  0.1 ],
                              [ 0.1 ,  0.4 ,  1.0 ,  0.8 ,  0.4 ,  0.1 ],
                              [ 0.1 ,  0.2 ,  0.2 ,  0.2 ,  0.2 ,  0.1 ],
                              [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ]])
    if axes == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    ax.imshow(current_data, interpolation='None', origin='lower')
    plt.xticks(np.arange(0,6,1),('left','','','','','right'), fontsize=20)
    plt.yticks(np.arange(0,6,1),('bottom','','','','','surface'), fontsize=20)
    return fig

if __name__ == "__main__":
    plot_tank()
