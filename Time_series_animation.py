import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque


class Time_series_animation():
    """
    A class used to create animations with matplotlib

    ...

    Parameters
    ----------
    reconstruction_errors: list or numpy.ndarray
        history of reconstruction errors
    threshold: list or numpy.ndarray
        history of threshold values
    class_labels: list or numpy.ndarray
        class labels denoting anomalies (1) and normal data (0)
    animation_window_size:
        how many points to show at once (default: 300)
    update_interval:
        defines the speed of the animation (default 25ms)
    start: int
        from which index to start the visualization (default: 0)
    hide_classes: boolean
        color reconstruction errors according to their classes (default: True)
    adjust_y_to_threshold: boolean 
        adjust the maximum value of the y axis  (default False)
    show_std: boolean
        showing the computed sliding-window variance of the reconstruction errors (default: True)
    std_window_size: int
        the size of the sliding window (default: 50)

    """

    def __init__(self, **kwargs):
        """Initialize Time_series_animation object
            **kwargs: see above
        """
        self.init_class_variables(kwargs)
        self.init_queues()
        self.init_colors()
        self.init_plot()

    def init_class_variables(self, kwargs):
        """Initialize Time_series_animation object's class variables according to user settings or to default values
            kwargs: contains the dictionary to set the class variables
        """
        keys = ['reconstruction_errors', 'threshold', 'class_labels', 'animation_window_size', 'update_interval', 'start', 'hide_classes', 'adjust_y_to_threshold', 'show_std', 'std_window_size']
        defaults = [[], [], [], 300, 25, 0, True, False, True, 50]
        for index, key in enumerate(keys):
            setattr(self, key, kwargs.get(key, defaults[index]))

    def init_plot(self):
        """Initialize plot
        """
        self.fig, self.ax = plt.subplots()

        if (self.hide_classes):
            self.line_0, = self.ax.plot(
                self.x, self.y, 'o', color='grey', markersize=1)
            self.line_1 = []
        else:
            self.line_0, = self.ax.plot(self.x, self.y, 'o', color='green', markersize=1)
            self.line_1, = self.ax.plot(self.x, self.y, 'o', color='red', markersize=1)
        self.th_line, = self.ax.plot(self.x, self.th, color='black', marker='.',
                            linestyle='-', linewidth=2, markersize=0.5)
        if (self.show_std):
            self.var_line, = self.ax.plot(self.x, self.std, color='blue', marker='.',
                            linestyle='-', linewidth=2, markersize=0.5)
        else:
            self.var_line = []

        self.animation = animation.FuncAnimation(fig=self.fig, func=self.update_animation, init_func=self.init_animation, interval=self.update_interval, blit=False, frames = len(self.reconstruction_errors)-self.start)
        plt.ylabel('reconstruction error')
        plt.xlabel('time step')
        if (self.hide_classes):
            legend = ['reconstruction error', 'threshold']
        else:
            legend = ['normal reconstruction error', 'anomaly reconstruction error', 'threshold']
        if (self.show_std):
            legend.append('sliding window std (window size=' + str(self.std_window_size) + ')')
        plt.legend(legend, loc='upper right', ncol=2)

    def init_queues(self):
        """Initialize data queues for the visualitation (of size animation_window_size)
        """
        self.x = deque(maxlen=self.animation_window_size)
        self.y = deque(maxlen=self.animation_window_size)
        self.th = deque(maxlen=self.animation_window_size)
        self.classes = deque(maxlen=self.animation_window_size)
        if (self.show_std):
            self.std = deque(maxlen=self.animation_window_size)
        else:
            self.std = [0]

    def update_queues(self, index):
        """Update queues to render new frames
            i: Frame index
        """
        if (len(self.x) > 0):
            self.x.append(self.x[-1]+1)
        else: 
            self.x.append(self.start)
        self.y.append(self.reconstruction_errors[self.start+index])
        self.th.append(self.threshold[self.start+index])
        self.classes.append(self.class_labels[self.start+index])
        if (self.show_std):
            variance_at_i = np.std(self.reconstruction_errors[max(0,self.start+index-self.std_window_size):self.start+index+1])
            self.std.append(variance_at_i)

    def init_colors(self):
        """Initialize anomaly colors
        """
        if (len(self.class_labels) == 0): 
            return
        else:
            self.class_labels = [v if self.class_labels[i] == 1 else -1 for i, v in enumerate(self.reconstruction_errors)]

    def init_animation(self):
        """Initialize the animation
        """
        self.update_queues(0)
        self.ax.set_xlim(self.start+0, self.animation_window_size)
        self.ax.set_ylim(0, 0.5)
        self.line_0.set_ydata(np.ma.array(self.y, mask=True))
        if (not self.hide_classes):
            self.line_0.set_ydata(np.ma.array(self.y, mask=True))
        return self.line_0, self.line_1, self.th_line, self.var_line

    def update_animation(self, index):
        """Update the animation, calles queue update
            index: Frame index
        """
        if (index+1 >= len(self.reconstruction_errors)-self.start):
            plt.close()
            return
        self.update_queues(index)
        self.line_0.set_xdata(self.x)
        self.line_0.set_ydata(self.y)

        if ((not self.hide_classes)):
            self.line_1.set_xdata(self.x)
            self.line_1.set_ydata(self.classes)

        self.th_line.set_xdata(self.x)
        self.th_line.set_ydata(self.th)
        
        if (self.show_std): 
            self.var_line.set_xdata(self.x)	
            self.var_line.set_ydata(self.std)
        if (self.adjust_y_to_threshold):
            self.ax.set_ylim(0, max(np.max(self.y), np.max(self.th), np.max(self.std))*1.2)
        else:
            self.ax.set_ylim(0, np.max(self.y)*1.2)

        self.ax.set_xlim(np.min(self.x), (np.min(self.x)+int(self.animation_window_size*1.4))) # adds some padding to the window size
        return self.line_0, self.line_1, self.th_line, self.var_line

    def save_to_file(self, file_name):
        """Export mp4 file
            file_name: custom file_name with mp4 extension
        """
        self.fig.set_figheight(4)
        self.fig.set_figwidth(10)
        self.fig.subplots_adjust(left=0.07, bottom=0.15, right=0.99, top=0.99, wspace=0, hspace=0)
        fps = math.ceil(1000.0/self.update_interval)
        self.animation.save(file_name,  writer='imagemagick', fps=fps, metadata=dict(artist='DoraSz'), dpi=100)
    
    def play(self):    
        """Start animation
        """    
        plt.show()

if __name__ == "__main__":
    pass
