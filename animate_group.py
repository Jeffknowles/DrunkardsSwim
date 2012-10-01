import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
import walkfunction as wlk
import plot_functions


FLOW_DYNAMICS = 'JEB'

t_noflow = 10 # in sec
t_flow = 50 # in sec

binsize = 0.1
current = 5 # in cm/s.


n_animals = 100

# generate data
track_data = {}
track_data['no_ll'] = []
for k in range(0, n_animals):
    ipos = np.array([np.random.uniform(0,.68, 1), np.random.uniform(0,.15,1), np.random.uniform(0,.15,1)]).flatten()
    data = wlk.randwalk(t_noflow, binsize, 0, ipos, latline = False, flow_version = FLOW_DYNAMICS)
    data = wlk.randwalk(t_flow, binsize, current, data, latline = False, flow_version = FLOW_DYNAMICS)
    data['track'] = data['track'] * 100
    track_data['no_ll'].append(data)

track_data['ll'] = []
for k in range(0, n_animals):
    ipos = np.array([np.random.uniform(0,.68, 1), np.random.uniform(0,.15,1), np.random.uniform(0,.15,1)]).flatten()
    data = wlk.randwalk(t_noflow, binsize, 0, ipos, latline = True, flow_version = FLOW_DYNAMICS)
    data = wlk.randwalk(t_flow, binsize, current, data, latline = True, flow_version = FLOW_DYNAMICS)
    data['track'] = data['track'] * 100
    track_data['ll'].append(data)

flow = np.ones(track_data['no_ll'][0]['track'].shape[0],'int') * current
flow[0: (float(t_noflow)/ binsize)] = 0

state_colors = ['blue', 'black', 'red']

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        self.fig = plt.figure(facecolor = 'w', figsize = [12, 6])
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        ax1 = self.fig.add_subplot(2, 1, 1, projection='3d')
        ax2 = self.fig.add_subplot(2, 1, 2, projection='3d')
        
        self.track_data = track_data
        
        self.t = np.arange(0,len(track_data['no_ll'][0]['track']))
        self.flow = flow[self.t]
        self.flow_flash = 0


        self.info1 = ax2.text2D(0.90,0.95,'State key',fontsize = 18,transform=ax2.transAxes)
        self.info2 = ax2.text2D(0.90,0.90,'------------',fontsize = 18, transform=ax2.transAxes)
        self.info2 = ax2.text2D(0.90,0.75,'Blue: rest',fontsize = 18, transform=ax2.transAxes)
        self.info3 = ax2.text2D(0.90,0.60,'Black: swim ', fontsize = 18, transform=ax2.transAxes)
        self.info4 = ax2.text2D(0.90,0.45,'Red: lat. line.', fontsize = 18, transform=ax2.transAxes)

        self.flowtexta1 = ax1.text2D(0.001, 0.65, 'Flow 0 cm/s', fontsize=18, transform=ax1.transAxes)
        self.flowtexta2 = ax2.text2D(0.001, 0.65, 'Flow 0 cm/s', fontsize=18, transform=ax2.transAxes)

        self.meantext1 = ax1.text2D(0.4, 0, 'X mean = 37.5 +/- 0', fontsize=18, color = 'green', transform=ax1.transAxes)
        self.meantext2 = ax2.text2D(0.4, 0, 'X mean = 37.5 +/- 0', fontsize=18, color = 'green', transform=ax2.transAxes)

        self.ax1mean = Line3D([37.5, 37.5], [0, 15], [0, 0], color = 'green', linewidth = 2)
        ax1.add_line(self.ax1mean)
        self.ax2mean = Line3D([37.5, 37.5], [0, 15], [0, 0], color = 'green', linewidth = 2)
        ax2.add_line(self.ax2mean)

        plot_functions.plot_tank(ax1)
        plot_functions.plot_tank(ax2)

        self.ax1_lines = []
        for animal in self.track_data['no_ll']:
            line_dict = {}
            line_dict['head'] = Line3D([], [], [], color='blue', marker='o', markeredgecolor='black')
            line_dict['tail'] = Line3D([], [], [], color='blue', linewidth=2)
            ax1.add_line(line_dict['head'])
            ax1.add_line(line_dict['tail'])
            self.ax1_lines.append(line_dict)

        self.ax2_lines = []
        for animal in self.track_data['ll']:
            line_dict = {}
            line_dict['head'] = Line3D([], [], [], color='blue', marker='o', markeredgecolor='black')
            line_dict['tail'] = Line3D([], [], [], color='blue', linewidth=2)
            ax2.add_line(line_dict['head'])
            ax2.add_line(line_dict['tail'])
            self.ax2_lines.append(line_dict)

        ax1.set_xlim(0, 68)
        ax1.set_ylim(0, 15)
        ax1.set_zlim(0, 15)
        ax1.set_title('Passive Model')
        ax1.set_aspect('equal')

        ax2.set_xlim(0, 68)
        ax2.set_ylim(0, 15)
        ax2.set_zlim(0, 15)
        ax2.set_title('Active Lateral Line')
        ax2.set_aspect('equal')
        
        ax1.set_xticks([0, 68])
        ax1.set_yticks([])
        ax1.set_zticks([0, 15])

        ax2.set_xticks([0, 68])
        ax2.set_yticks([])
        ax2.set_zticks([0, 15])

        plt.rc('xtick', labelsize=20)

        animation.TimedAnimation.__init__(self, self.fig, interval=10, blit=False)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_len = 10
        head_slice = (self.t > self.t[i] - 7.0) & (self.t < self.t[i])

        self._drawn_artists = []

        for k, line_dict in enumerate(self.ax1_lines):
            line_dict['tail'].set_data(self.track_data['no_ll'][k]['track'][head_slice, 0], self.track_data['no_ll'][k]['track'][head_slice, 1])
            line_dict['tail'].set_3d_properties(self.track_data['no_ll'][k]['track'][head_slice, 2])
            line_dict['tail'].set_color(state_colors[int(self.track_data['no_ll'][k]['state'][head])])
            self._drawn_artists.append(line_dict['tail'])
            line_dict['head'].set_data(self.track_data['no_ll'][k]['track'][head, 0], self.track_data['no_ll'][k]['track'][head, 1])
            line_dict['head'].set_3d_properties(self.track_data['no_ll'][k]['track'][head, 2])
            line_dict['head'].set_color(state_colors[int(self.track_data['no_ll'][k]['state'][head])])
            self._drawn_artists.append(line_dict['head'])
            
            
        for k, line_dict in enumerate(self.ax2_lines):
            line_dict['tail'].set_data(self.track_data['ll'][k]['track'][head_slice, 0], self.track_data['ll'][k]['track'][head_slice, 1])
            line_dict['tail'].set_3d_properties(self.track_data['ll'][k]['track'][head_slice, 2])
            line_dict['tail'].set_color(state_colors[int(self.track_data['ll'][k]['state'][head])])
            self._drawn_artists.append(line_dict['tail'])
            line_dict['head'].set_data(self.track_data['ll'][k]['track'][head, 0], self.track_data['ll'][k]['track'][head, 1])
            line_dict['head'].set_3d_properties(self.track_data['ll'][k]['track'][head, 2])
            line_dict['head'].set_color(state_colors[int(self.track_data['ll'][k]['state'][head])])
            self._drawn_artists.append(line_dict['head'])


        mean_x = np.mean([animal['track'][head,0] for animal in track_data['no_ll']])
        std_x = np.std([animal['track'][head,0] for animal in track_data['no_ll']])
        self.meantext1.set_text('Xmean = ' + str(round(mean_x)) + ' +/- ' + str(int(round(std_x))))
        self.ax1mean.set_data([mean_x, mean_x], [0, 15])
        self.ax1mean.set_3d_properties([0, 0])
        self._drawn_artists.append(self.ax1mean)
        
        mean_x = np.mean([animal['track'][head,0] for animal in track_data['ll']])
        std_x = np.std([animal['track'][head,0] for animal in track_data['ll']])
        self.meantext2.set_text('Xmean = ' + str(round(mean_x)) + ' +/- ' + str(int(round(std_x))))
        self.ax2mean.set_data([mean_x, mean_x], [0, 15])
        self.ax2mean.set_3d_properties([0, 0])
        self._drawn_artists.append(self.ax2mean)

        if self.flow[head] > 0 and self.flow_flash < 20: 
            self.flow_flash += 1
            if isodd(self.flow_flash):
                self.flowtexta1.set_text('')
                self.flowtexta2.set_text('')
            else:
                self.flowtexta1.set_text('Flow ' + str(self.flow[head]) + ' cm/s')
                self.flowtexta2.set_text('Flow ' + str(self.flow[head]) + ' cm/s')
        else:
            self.flowtexta1.set_text('Flow ' + str(self.flow[head]) + ' cm/s')
            self.flowtexta2.set_text('Flow ' + str(self.flow[head]) + ' cm/s')


        

    def new_frame_seq(self):
        return iter(range(self.t.size))

    # def _init_draw(self):

    #     lines =  [self.line1, self.line1a, self.line1e,
    #               self.line2, self.line2a, self.line2e]
    #     for l in lines:
    #         l.set_data([], [])


def isodd(num):
    return num & 1 and True or False

if __name__ == "__main__":
    ani = SubplotAnimation()
    ani.save('group2g.mp4', fps=20, codec='mpeg4', clear_temp=True, frame_prefix='_tmp')
    #plt.show()


