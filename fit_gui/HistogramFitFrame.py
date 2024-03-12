import numpy as np
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.optimize as opt

if __name__ == '__main__':
    import FitElementFrame
else:
    from . import FitElementFrame 

#TODO: ability to fix some parameters
#TODO: ability to only fix some region of the histogram
#TODO: use root for fitting
class HistogramFitFrame(ttk.Frame):
    def __init__(self, parent, data):
        '''
        data: array of floats to make histogram of
        '''
        super().__init__(parent)
        self.data = data

        matplotlib.use('TkAgg')

        self.figure = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)
        self.canvas.mpl_connect('button_press_event', self.on_plot_clicked)
        self.hist_ax = self.figure.add_subplot(211)
        self.residual_ax = self.figure.add_subplot(212)

        ttk.Label(self, text='# bins').grid(row=0, column=0, sticky='e')
        self.bins_entry = ttk.Entry(self)
        self.bins_entry.insert(0, '100')
        self.bins_entry.grid(row=0, column=1, sticky='w')
        self.bins_entry.bind("<FocusOut>", lambda e: self.rebin())

        fit_button = ttk.Button(self,text='Fit', command = self.fit_button_clicked)
        fit_button.grid(row=0, column=2, sticky='e')

        self.elements_frame = ttk.Frame(self)
        self.elements_frame.grid(row=1, column=3, rowspan=1, sticky="nsew")
        
        self.elements=[]

        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=3)
        values=['Gaussian', 'Linear', 'Bragg', 'Bragg w/ Diffusion']
        self.add_element_combo = ttk.Combobox(top_frame, 
                                              values=values)
        self.add_element_combo.current(0)
        self.add_element_combo.pack(side=tk.LEFT)
        ttk.Button(top_frame, text='add element',
                   command=self.add_element_clicked).pack(side=tk.RIGHT)
        self.rebin()

    def rebin(self):
        self.num_bins = int(self.bins_entry.get())
        self.hist, self.bins = np.histogram(self.data, self.num_bins)
        self.bin_centers = (self.bins[1:] + self.bins[:-1])/2
        self.update_hist()

    def get_bin_size(self):
        return self.bins[1] - self.bins[0]

    def update_hist(self):
        self.hist_ax.clear()
        bin_size = self.get_bin_size()
        self.hist_ax.bar(self.bin_centers, self.hist, width=bin_size)
        if len(self.elements) > 0:
            #draw guess and fit lines
            guess = np.zeros(self.num_bins)
            fit = np.zeros(self.num_bins)
            for element in self.elements:
                guess += element.evaluate(self.bin_centers, element.get_guess_params())
                fit += element.evaluate(self.bin_centers, element.get_fit_params())
            guess *= bin_size
            fit *= bin_size
            self.hist_ax.plot(self.bin_centers, guess, color='orange', label='guess')
            self.hist_ax.plot(self.bin_centers, fit, color='r', label='fit')
            self.hist_ax.legend()
            #draw residuals
            self.residual_ax.clear()
            self.residual_ax.bar(self.bin_centers, self.hist - fit, width=bin_size)
        self.residual_ax.set_title('residuals (actual - fit)')
        self.canvas.draw()

    def add_element_clicked(self):
        if self.add_element_combo.get() == 'Gaussian':
            to_add = FitElementFrame.Gaussian(self.elements_frame, self)
        elif self.add_element_combo.get() == 'Linear':
            to_add = FitElementFrame.Linear(self.elements_frame, self)
        elif self.add_element_combo.get() == 'Bragg':
            to_add = FitElementFrame.Bragg(self.elements_frame, self)
        elif self.add_element_combo.get() == 'Bragg w/ Diffusion':
            to_add = FitElementFrame.BraggWDiffusion(self.elements_frame, self)
        self.elements.append(to_add)
        to_add.pack(side=tk.TOP)
        self.update_hist()

    def remove_element(self, element):
        self.elements.remove(element)
        self.update_hist()

    def on_plot_clicked(self, event):
        self.last_clicked_point = (event.xdata, event.ydata)

    def fit_button_clicked(self):
        xs = self.bin_centers
        #get initial guess from all elements
        init_guess = []
        for e in self.elements:
            for v in e.get_guess_params():
                init_guess.append(v)
        #make a function which return values for all current entries
        def f(x, *params):
            to_return = np.zeros(len(xs))
            i = 0
            for e in self.elements:
                params_to_pass = []
                while len(params_to_pass) < len(e.guess_entries):
                    params_to_pass.append(params[i])
                    i += 1
                to_return += e.evaluate(x, params_to_pass)
            return to_return
        #do the fit
        popt, pcov, infodict, mesg, ier =  opt.curve_fit(f, xs, self.hist/self.get_bin_size(), init_guess, full_output=True)
        print(mesg)
        print(np.sqrt(np.diag(pcov)))
        i = 0
        for e in self.elements:
            for n in range(len(e.guess_entries)):
                e.set_fit_param(n, popt[i])
                i += 1
        self.update_hist()
        



if __name__ == '__main__':
    import numpy.random as random
    '''data = np.concatenate(\
        [random.normal(-0.5, 0.3,1000),\
         np.random.normal(0.5,0.2, 300), \
         np.random.normal(3, 1, 3000),
         np.random.uniform(-5,5,500)])#b=50, m=0'''

    file_path = tk.filedialog.askopenfilename()
    data = np.load(file_path)
    include_all_data = True
    print('total events in file = %d'%len(data))
    if not include_all_data:
        min_val, max_val = 4e5, 6e5
        data = data[np.logical_and(data>min_val, data<max_val)]
        print('events after applying cut = %d'%len(data))

    root = tk.Tk()
    frame = HistogramFitFrame(root, data)
    frame.grid()
    root.mainloop()