import numpy as np
import scipy.integrate as integrate
import scipy.ndimage

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class FitElementFrame(tk.Frame):
    def __init__(self, master, hist_fit_frame, param_names:list, 
                 default_values:list):
        super().__init__(master, highlightbackground="black",
                         highlightthickness=1)
        self.hist_fit_frame = hist_fit_frame

        self.guess_entries = []
        self.fit_entries = []
        self.lowerbound_entries = []
        self.upperbound_entries = []
        ttk.Button(self, text='X', command=self.X_clicked).grid(column=0, row=0)
        ttk.Label(self, text='init guess').grid(row=0, column=1)
        ttk.Label(self, text='parameter fit').grid(row=0, column=2)
        ttk.Label(self, text='lower/upper bound').grid(row=0, column=3, columnspan=2)
        self.next_row = 1
        for name, val in zip(param_names, default_values):
            self.add_param_entry(name, val)

    def add_param_entry(self, name, default_val):
        ttk.Label(self, text=name).grid(row=self.next_row, column=0)
        new_guess_entry = ttk.Entry(self)
        new_guess_entry.insert(0, str(default_val))
        new_guess_entry.grid(row=self.next_row, column=1)
        new_guess_entry.bind("<FocusOut>", lambda e: self.hist_fit_frame.update_hist())
        self.guess_entries.append(new_guess_entry)
        
        new_fit_entry = ttk.Entry(self)
        new_fit_entry.grid(row=self.next_row, column=2)
        new_fit_entry.bind("<FocusOut>", lambda e: self.hist_fit_frame.update_hist())
        self.fit_entries.append(new_fit_entry)
        self.set_fit_param(-1, default_val)
        
        self.lowerbound_entries.append(ttk.Entry(self))
        self.lowerbound_entries[-1].grid(row=self.next_row, column=3)
        self.upperbound_entries.append(ttk.Entry(self))
        self.upperbound_entries[-1].grid(row=self.next_row, column=4)
        self.next_row += 1

    def get_guess_params(self):
        '''
        entries_dict = self.guess_entires or fit_entries
        '''
        return [float(e.get()) for e in self.guess_entries]
    
    def get_fit_params(self):
        return [float(e.get()) for e in self.fit_entries]
    
    def set_fit_param(self, index, value):
        self.fit_entries[index].delete(0, tk.END)
        self.fit_entries[index].insert(0, value)

        
    def update_guess(self, index, new_value):
        widget = self.guess_entries[index]
        widget.delete(0, tk.END)
        widget.insert(0, str(new_value))
    
    def X_clicked(self):
        self.hist_fit_frame.remove_element(self)
        self.destroy()

    def evaluate(self, xs:np.array, params:np.array):
        '''
        Should return density distribution (eg counts/dx, etc) at each
        point listed in xs. This way parameters don't need to be rescaled when
        histogram is rebinned.
        '''
        return
    

class Gaussian(FitElementFrame):
    '''
    magnitude * exp(-0.5*((x-mu)/sigma)^2)/(sigma sqrt(2 pi))

    parameter order for evaluate: mu, sigma, magnitude
    '''
    def __init__(self, master, hist_fit_frame):
        super().__init__(master, hist_fit_frame, ['mu', 'sigma', 'magnitude'], [0,1,0])
        help_button = ttk.Button(self, text='?', command=self.show_help)
        help_button.grid(row=self.next_row, column=2)
        
        peak_button = ttk.Button(self, text='peak', command=self.peak_clicked)
        peak_button.grid(row=self.next_row, column=0)
        self.peak_clicked = False
        
        fwhm_button = ttk.Button(self, text='FWHM', command=self.fwhm_clicked)
        fwhm_button.grid(row=self.next_row, column=1)
        self.fwhm_clicked = False

    def show_help(self):
        help_string = \
            '''
            Click the top of the peak you want to fit and then click "peak" button.
            Then click FWHM guess on the graph, followed by "FWHM" button".
            Initial guesses for parameters will then be autopopulated.
            '''
        messagebox.showinfo('help', help_string)

    def fwhm_clicked(self):
        self.fwhm_point = self.hist_fit_frame.last_clicked_point
        self.fwhm_clicked = True
        if self.peak_clicked:
            self.update_guess_from_clicks()
    
    def peak_clicked(self):
        self.peak_point = self.hist_fit_frame.last_clicked_point
        self.peak_clicked = True
        if self.fwhm_clicked:
            self.update_guess_from_clicks()

    def update_guess_from_clicks(self):
        fwhm = 2*np.abs(self.peak_point[0] - self.fwhm_point[0])
        sigma = fwhm/2/np.sqrt(2*np.log(2))
        mu = self.peak_point[0]
        magnitude = self.peak_point[1]*sigma*np.sqrt(2*np.pi)/self.hist_fit_frame.get_bin_size()
        self.update_guess(1, sigma)
        self.update_guess(0, mu)
        self.update_guess(2, magnitude)
        self.hist_fit_frame.update_hist()


    def evaluate(self, xs, params):
        mu, sigma, A = params
        return A/np.sqrt(2*np.pi)/sigma*np.exp(-0.5*((xs - mu)/sigma)**2)
            
class Linear(FitElementFrame):
    '''
    slope*x + offset
    '''
    def __init__(self, master, hist_fit_frame):
        super().__init__(master, hist_fit_frame, ['slope', 'offset'], [0,0])
    
    def evaluate(self, xs, params):
        m, b = params
        return m*xs + b

class Bragg(FitElementFrame):
    '''
    x0: point charged particle is generated
    E0: charged particle initial energy
    c1, c2, c3: Parameters for fit

    beta^2 = c3 * E
    '''
    def __init__(self, master, hist_fit_frame):
        super().__init__(master, hist_fit_frame, ['x0','E0','c1', 'c2', 'c3'], [0,0.001,1,50000,.0001])
        ttk.Label(self, text='direction').grid(row=self.next_row, column=0)
        self.direction_combobox = ttk.Combobox(self, values=['left', 'right'])
        self.direction_combobox.set('right')
        self.direction_combobox.grid(row=self.next_row, column=1)
        self.direction_combobox.bind("<<ComboboxSelected>>", lambda e: self.hist_fit_frame.update_hist())
        self.next_row += 1

    def evaluate(self, xs: np.array, params):
        x0, E0, c1, c2, c3 = params
        direction = self.direction_combobox.get()
        if direction == 'right':
            xs_for_int = np.concatenate([[x0], xs[xs>=x0]])
        else:
            xs_for_int = np.flip(np.concatenate([xs[xs<=x0], [x0]]))
            c1=-c1

        if len(xs_for_int) == 1:
            return np.zeros(len(xs))

        dEdx = lambda E: -c1*(np.log(c2*c3*E/(1-c3*E))/c3/E - 1)
        Es = np.squeeze(integrate.odeint(lambda E, x: dEdx(E), E0, xs_for_int))

        to_return = np.zeros(len(xs))
        if direction == 'right':
            to_return[xs>=x0] = -dEdx(Es[1:])
        else:
            Es = np.flip(Es)
            to_return[xs<=x0] = dEdx(Es[1:])
        return to_return
        
class BraggWDiffusion(Bragg):
    def __init__(self, master, hist_fit_frame):
        super().__init__(master, hist_fit_frame)
        self.add_param_entry('sigma', 1)
    
    def evaluate(self, xs, params):
        x0, E0, c1, c2, c3, sigma = params
        #should consider oversampling if needed
        no_diff = super().evaluate(xs, params[0:-1])
        dx = xs[1] - xs[0]
        sigma_in_bins = sigma / dx
        return scipy.ndimage.gaussian_filter1d(no_diff, sigma_in_bins, mode='nearest')

class ProtonAlpha(BraggWDiffusion):
    '''
    This class assumes alpha stopping distance is very small
    '''
    def __init__(self, master, hist_fit_frame):
        super().__init__(master, hist_fit_frame)
        self.add_param_entry('Ealpha', 0)

        peak_button = ttk.Button(self, text='peak', command=self.peak_clicked)
        peak_button.grid(row=self.next_row, column=0)
        self.peak_clicked = False
        
        fwhm_button = ttk.Button(self, text='FWHM', command=self.fwhm_clicked)
        fwhm_button.grid(row=self.next_row, column=1)
        self.fwhm_clicked = False
    
    def evaluate(self, xs, params):
        x0, Eproton, c1, c2, c3, sigma, Ealpha = params
        #should consider oversampling if needed
        proton = super().evaluate(xs, params[0:-1])
        alpha =  Ealpha/np.sqrt(2*np.pi)/sigma*np.exp(-0.5*((xs - x0)/sigma)**2)
        return proton + alpha
    
    def fwhm_clicked(self):
        self.fwhm_point = self.hist_fit_frame.last_clicked_point
        self.fwhm_clicked = True
        if self.peak_clicked:
            self.update_guess_from_clicks()
    
    def peak_clicked(self):
        self.peak_point = self.hist_fit_frame.last_clicked_point
        self.peak_clicked = True
        if self.fwhm_clicked:
            self.update_guess_from_clicks()

    def update_guess_from_clicks(self):
        #update gaussian for alpha energy deposition
        fwhm = 2*np.abs(self.peak_point[0] - self.fwhm_point[0])
        sigma = fwhm/2/np.sqrt(2*np.log(2))
        x0 = self.peak_point[0]
        Ealpha = self.peak_point[1]*sigma*np.sqrt(2*np.pi)/self.hist_fit_frame.get_bin_size()
        self.update_guess(5, sigma)
        self.update_guess(0, x0)
        self.update_guess(6, Ealpha)
        #update proton guess to account for the rest of the energy
        xs = self.hist_fit_frame.xs
        ys = self.hist_fit_frame.ys
        total_energy = np.trapz(ys, xs)
        Eproton = total_energy - Ealpha
        self.update_guess(1, Eproton)
        #assume a small initial beta
        beta0 = 0.01
        c2 = self.get_guess_params()[3]
        c3 = beta0**2/Eproton
        self.update_guess(4, c3)
        dEdx_guess = Eproton/np.abs(xs[-1] - xs[0])
        c1 = dEdx_guess*(beta0**2)/(np.log(c2*beta0**2) - beta0**2)
        self.update_guess(2, 2*c1)#factor of 2 is empirical
        self.hist_fit_frame.update_hist()