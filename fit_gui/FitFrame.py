import numpy as np
import tkinter as tk
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
class FitFrame(ttk.Frame):
    def __init__(self, parent, xs, ys):
        super().__init__(parent)
        self.xs = xs
        self.ys = ys

        matplotlib.use('TkAgg')

        self.figure = plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3)
        self.canvas.mpl_connect('button_press_event', self.on_plot_clicked)
        self.data_ax = self.figure.add_subplot(211)
        self.residual_ax = self.figure.add_subplot(212)

        fit_button = ttk.Button(self,text='Fit', command = self.fit_button_clicked)
        fit_button.grid(row=0, column=2, sticky='e')

        self.elements_frame = ttk.Frame(self)
        self.elements_frame.grid(row=1, column=3, rowspan=1, sticky="nsew")
        
        self.elements=[]

        top_frame = ttk.Frame(self)
        top_frame.grid(row=0, column=3)
        values=['Gaussian', 'Linear', 'Bragg', 'Bragg w/ Diffusion',
                'Proton + Alpha']
        self.add_element_combo = ttk.Combobox(top_frame, 
                                              values=values)
        self.add_element_combo.current(0)
        self.add_element_combo.pack(side=tk.LEFT)
        ttk.Button(top_frame, text='add element',
                   command=self.add_element_clicked).pack(side=tk.RIGHT)
        
        self.draw_plot()
        plt.autoscale(False)

    #define some things so that FitElements I wrote for histograms will still work
    #TODO: revisit how I structured this code.
    def get_bin_size(self):
        return 1 
    def update_hist(self):
        self.draw_plot()

    def draw_plot(self):
        self.data_ax.clear()
        self.data_ax.scatter(self.xs, self.ys)
        if len(self.elements) > 0:
            #draw guess and fit lines
            guess = np.zeros(len(self.xs))
            fit = np.zeros(len(self.xs))
            for element in self.elements:
                guess += element.evaluate(self.xs, element.get_guess_params())
                fit += element.evaluate(self.xs, element.get_fit_params())
            self.data_ax.plot(self.xs, guess, color='orange', label='guess')
            self.data_ax.plot(self.xs, fit, color='r', label='fit')
            self.data_ax.legend()
            #draw residuals
            self.residual_ax.clear()
            self.residual_ax.scatter(self.xs, self.ys - fit)
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
        elif self.add_element_combo.get() == 'Proton + Alpha':
            to_add = FitElementFrame.ProtonAlpha(self.elements_frame, self)
        self.elements.append(to_add)
        to_add.pack(side=tk.TOP)
        self.draw_plot()

    def remove_element(self, element):
        self.elements.remove(element)
        self.draw_plot()

    def on_plot_clicked(self, event):
        self.last_clicked_point = (event.xdata, event.ydata)

    def fit_button_clicked(self):
        xs = self.xs
        #get initial guess from all elements
        init_guess = []
        for e in self.elements:
            for v in e.get_guess_params():
                init_guess.append(v)
        #make a function which return values for all current entries
        def f(x, *params):
            to_return = np.zeros(len(self.xs))
            i = 0
            for e in self.elements:
                params_to_pass = []
                while len(params_to_pass) < len(e.guess_entries):
                    params_to_pass.append(params[i])
                    i += 1
                to_return += e.evaluate(x, params_to_pass)
            return to_return
        #get bounds to impose
        bounds = ([],[])
        for e in self.elements:
            for lb_entry, ub_entry in zip(e.lowerbound_entries, e.upperbound_entries):
                lb_str = lb_entry.get()
                if len(lb_str) == 0:
                    bounds[0].append(-np.inf)
                else:
                    bounds[0].append(float(lb_str))
                ub_str = ub_entry.get()
                if len(ub_str) == 0:
                    bounds[1].append(np.inf)
                else:
                    bounds[1].append(float(ub_str))
        #do the fit
        popt, pcov =  opt.curve_fit(f, self.xs, self.ys,
                                    init_guess, bounds=bounds)
        #print(mesg)
        print(np.sqrt(np.diag(pcov)))
        i = 0
        for e in self.elements:
            for n in range(len(e.guess_entries)):
                e.set_fit_param(n, popt[i])
                i += 1
        self.draw_plot()

if __name__ == '__main__':
    import numpy.random as random
    xs = np.linspace(0,10)
    ys = np.sin(xs)

    root = tk.Tk()
    frame = FitFrame(root, xs, ys)
    frame.grid()
    root.mainloop()