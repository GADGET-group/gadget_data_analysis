import tkinter as tk
from tkinter import ttk
import gadget_widgets

import matplotlib.pyplot as plt
import numpy as np

import scipy.optimize as opt
import scipy.stats as stats

class EnergySpectrumFrame(ttk.Frame):
    def __init__(self, parent, run_data):
        super().__init__(parent)
        self.run_data = run_data
        #show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')
        #add GUI elements
        num_cols=1
        current_row = 0
        
        self.hist_settings_frame = ttk.Labelframe(self, text='histogram settings')
        self.hist_settings_frame.grid(row=current_row, column=0)
        current_row += 1
        self.num_bins_label = ttk.Label(self.hist_settings_frame, text='num bins:')
        self.num_bins_label.grid(row=0, column=0, sticky=tk.E)
        self.num_bins_entry = gadget_widgets.GEntry(self.hist_settings_frame, default_text='Enter # of Bins')
        self.num_bins_entry.grid(row=0, column=1, columnspan=num_cols)
        self.units_select_label = ttk.Label(self.hist_settings_frame, text='energy units:')
        self.units_select_label.grid(row=1, column=0, sticky=tk.E)
        self.units_combobox = ttk.Combobox(self.hist_settings_frame, values=['adc counts', 'MeV'])
        self.units_combobox.set('adc counts')
        self.units_combobox.grid(row=1, column=1)
        self.generate_spectrum_button = \
                    ttk.Button(self.hist_settings_frame, text="Show Spectrum", command=self.plot_spectrum)
        self.generate_spectrum_button.grid(row=2, column=0, columnspan=2)
        
        self.quick_fit_frame = ttk.Labelframe(self, text='quick fit')
        self.quick_fit_frame.grid(row=current_row, column=0)
        current_row += 1
        self.low_cut_entry = gadget_widgets.GEntry(self.quick_fit_frame, default_text='Low Cut Value')
        self.low_cut_entry.grid(row=0, column=0)
        self.high_cut_entry = gadget_widgets.GEntry(self.quick_fit_frame, default_text='High Cut Value')
        self.high_cut_entry.grid(row=0, column=1)
        self.show_cut_button = ttk.Button(self.quick_fit_frame, text='Show Cut Range', command=self.show_cut)
        self.show_cut_button.grid(row=1, column=0, columnspan=2)
        self.quick_fit_gaus_button = ttk.Button(self.quick_fit_frame, text="Quick Gaussian Fit",
                                command=self.quick_fit_gaussian)
        self.quick_fit_gaus_button.grid(row=2, column=0, columnspan=2)
        current_row += 1

        self.multi_peak_fit_frame = ttk.Labelframe(self, text='multi-peak fit')
        self.multi_peak_fit_frame.grid(row=current_row)
        current_row += 1
        self.multi_fit_button = ttk.Button(self.multi_peak_fit_frame,
                                 text="Initial Guesses for Multi-peak Fit",
                                 command=self.multi_fit_init_guess)
        self.multi_fit_button.grid(row=0)
        self.multi_fit_params_entry = gadget_widgets.GEntry(self.multi_peak_fit_frame, 
                'Paste Fit Parameters | Use * in Front of Param to Fix Value', width=42)
        self.multi_fit_params_entry.grid(row=1)
        self.multi_fit_from_params_button = ttk.Button(self.multi_peak_fit_frame, text="Multi-peak Fit from Params",
                 command=self.multi_fit_from_params)
        self.multi_fit_from_params_button.grid(row=2)

    def get_dataset(self):
        if self.units_combobox.get() == 'MeV':
            return self.run_data.total_energy_MeV
        elif self.units_combobox.get() == 'adc counts':
            return self.run_data.total_energy
        else:
            assert False

    def plot_1d_hist(self, fig_name):
        '''
        Shows a 1d histogram using the current settings.
        Methods calling this function should call plt.show() after this function call,
        and making any other desired modifications to the figure.
        '''
        num_bins = int(self.num_bins_entry.get())
        plt.figure(fig_name, clear=True)
        if self.units_combobox.get() == 'MeV':
            plt.xlabel(f'Energy (MeV)', fontdict = {'fontsize' : 20})
        elif self.units_combobox.get() == 'adc counts':
            plt.xlabel('Integrated Charge (adc counts)', fontdict = {'fontsize' : 20})
        to_plot = self.get_dataset()
        plt.hist(to_plot, bins=num_bins)
        plt.rcParams['figure.figsize'] = [10, 10]
        plt.title(f'Energy Spectrum', fontdict = {'fontsize' : 20})
        plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
        #plt.show()

    def plot_spectrum(self):
        self.plot_1d_hist('Energy Spectrum')
        plt.show()

    def show_cut(self):
        #show basic plot
        self.plot_1d_hist('Cut Range')
        #get number of counts in range
        low_cut = float(self.low_cut_entry.get())
        high_cut = float(self.high_cut_entry.get())
        dataset = self.get_dataset()
        trimmed_hist = dataset[np.logical_and(dataset>=low_cut, dataset<=high_cut)]
        plt.axvline(low_cut, color='red', linestyle='dashed', linewidth=2)
        plt.axvline(high_cut, color='red', linestyle='dashed', linewidth=2)
        plt.title(f'Number of Counts in Cut: {len(trimmed_hist)}',fontdict = {'fontsize' : 20})
        plt.show()

    def quick_fit_gaussian(self):
        def gaussian(x, amplitude, mu, sigma):
            return amplitude * np.exp(-(x - mu)**2 / (2 * sigma**2))
        low_cut_value = float(self.low_cut_entry.get())
        high_cut_value = float(self.high_cut_entry.get())
        dataset = self.get_dataset()
        num_bins = int(self.num_bins_entry.get())
        hist, bins = np.histogram(dataset, bins=num_bins, range=(low_cut_value, high_cut_value))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        #fit curve
        popt, pcov = opt.curve_fit(gaussian, bin_centers, hist, p0=[1, np.mean(dataset), np.std(dataset)], maxfev=800000)
        amplitude, mu, sigma = popt
        # Calculate chi-squared and p-value
        residuals = hist - gaussian(bin_centers, amplitude, mu, sigma)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((hist - np.mean(hist))**2)
        r_squared = 1 - (ss_res / ss_tot)
        chi_squared = ss_res / (num_bins - 3)
        dof = num_bins - 3
        chi_squared_dof = chi_squared / dof
        p_value = 1 - stats.chi2.cdf(chi_squared, dof)

        # Plot the histogram with the fit
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True, 
                               num='Quick Fit', clear=True)
        ax[0].hist(dataset, bins=num_bins, range=(low_cut_value, high_cut_value), histtype='step', color='blue', label='Data')
        x_fit = np.linspace(low_cut_value, high_cut_value, 100)
        ax[0].plot(x_fit, gaussian(x_fit, amplitude, mu, sigma), 'r-', label='Fit')
        ax[0].legend()
        ax[0].set_ylabel('Counts')

        # Plot the residuals
        ax[1].plot(bin_centers, residuals, 'b-', label='Residuals')
        ax[1].axhline(0, color='black', lw=1)
        ax[1].set_xlabel('Energy')
        ax[1].set_ylabel('Residuals')
        ax[1].legend()
        plt.tight_layout()

        # Display the fit parameters on the plot
        text = f'Chi-squared: {chi_squared:.2f}\nDegrees of Freedom: {dof}\nChi-squared per DOF: {chi_squared_dof:.2f}\np-value: {p_value:.2f}\nAmplitude: {amplitude:.2f}\nMu: {mu:.2f}\nSigma: {sigma:.2f}'
        ax[0].text(0.05, 0.95, text, transform=ax[0].transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        plt.show()

    #HACK: clean this function up
    def multi_fit_init_guess(self):
        #TODO: break this out into it's own gui
        '''
        Gets initial guesses, and then tries fitting guess
        '''
        #global peaks, peak_active, peak_data, peak_handle, peak_info

        dataset = self.get_dataset()

        # plot the histogram
        num_bins = int(self.num_bins_entry.get())
        fig, ax = plt.subplots()
        dataset = self.get_dataset()
        n, bins, patches = ax.hist(dataset, bins=num_bins)
        y_hist, x_hist = np.histogram(dataset, bins=num_bins)
        x_hist = (x_hist[:-1] + x_hist[1:]) / 2
        
        self.peak_handle, = plt.plot([], [], 'o', color='black', markersize=10, alpha=0.7)

        # keep track of the last left-click point
        last_left_click = None

        def onclick(event):
            #global peak_active, peak_handle, peak_info, horizontal_line
            if event.button == 1:  # Left mouse button
                x, y = event.xdata, event.ydata
                plt.plot(x, y, 'ro', markersize=10)
                plt.axvline(x, color='r', linestyle='--')
                plt.draw()
                self.peak_active = x

            elif event.button == 3:  # Right mouse button
                if self.peak_active is not None:
                    x, y = event.xdata, event.ydata
                    plt.plot(x, y, 'go', markersize=10)
                    plt.draw()

                    idx = np.argmin(np.abs(x_hist - self.peak_active))
                    mu = self.peak_active
                    sigma = np.abs(x - self.peak_active)
                    amp = y_hist[idx] * np.sqrt(2 * np.pi) * sigma
                    self.peak_info.extend([amp, mu, sigma, 1])

                    horizontal_line, = plt.plot([self.peak_active, x], [y, y], color='green', linestyle='--')

                    self.peak_active = None
                    plt.draw()

        # initialize peak detection variables
        self.peaks = []
        self.peak_data = []
        self.peak_active = None
        self.peak_info = []

        # connect the click event to the plot
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

        title1 = "Left Click to Select Peak Amp and Mu"
        title2 = "\nRight Click to Select Peak Sigma"

        # Calculate the position for each part of the title
        x1, y1 = 0.5, 1.10
        x2, y2 = 0.5, 1.05

        # Set the title using ax.annotate() and the ax.transAxes transform
        ax.annotate(title1, (x1, y1), xycoords='axes fraction', fontsize=12, color='red', ha='center', va='center')
        ax.annotate(title2, (x2, y2), xycoords='axes fraction', fontsize=12, color='green', ha='center', va='center')

        # show the plot
        plt.show()

        # Send peak_info for fitting when the plot is closed
        #print('INITIAL GUESSES:\n',peak_info)

        # Print initial guesses
        print("INITIAL GUESSES:")
        for i in range(0, len(self.peak_info), 4):
            print(f"Peak {i//4 + 1}: Amp={self.peak_info[i]}, Mu={self.peak_info[i+1]}, Sigma={self.peak_info[i+2]}, Lambda={self.peak_info[i+3]}")


        self.fit_multi_peaks(num_bins, x_hist, y_hist)
        
    #HACK: clean this function up
    def fit_multi_peaks(self, num_bins, x_hist, y_hist):
        #from scipy.special import erfc
        from scipy.special import erfcx
        from scipy.optimize import curve_fit
        from scipy.stats import chisquare, chi2
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea


        def safe_exp(x, min_exp_arg=None, max_exp_arg=None):
            min_exp_arg = min_exp_arg if min_exp_arg is not None else -np.inf
            max_exp_arg = max_exp_arg if max_exp_arg is not None else np.finfo(np.float64).maxexp - 10
            return np.exp(np.clip(x, min_exp_arg, max_exp_arg))

        
        def emg_stable(x, amplitude, mu, sigma, lambda_):
            exp_arg = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
            erfc_arg = (mu + lambda_ * sigma**2 - x) / (np.sqrt(2) * sigma)
            #print("lambda_: ", lambda_)
            #print("mu: ", mu)
            #print("sigma: ", sigma)
            #print("x: ", x)
            return 0.5 * amplitude * lambda_ * safe_exp(exp_arg - erfc_arg**2) * erfcx(erfc_arg)

        def composite_emg(x, *params):
            result = np.zeros_like(x)
            for i in range(0, len(params), 4):
                result += emg_stable(x, *params[i:i + 4])
            return result


        # Set the threshold for y_hist, adjust it based on your specific requirements
        y_hist_threshold = 1e5

        # Filter the data based on the threshold
        valid_indices = y_hist < y_hist_threshold
        filtered_x_hist = x_hist[valid_indices]
        filtered_y_hist = y_hist[valid_indices]


        # Fit the composite EMG function to the data
        popt, pcov = curve_fit(composite_emg, filtered_x_hist, filtered_y_hist, p0=self.peak_info, maxfev=1000000)
        fitted_emg = composite_emg(filtered_x_hist, *popt)
        #print('FINAL FIT PARAMETERS:', [*popt])
        # Print final fit parameters
        print("FINAL FIT PARAMETERS:")
        for i in range(0, len(popt), 4):
            print(f"Peak {i//4 + 1}: Amp={self.peak_info[i]}, Mu={self.peak_info[i+1]}, Sigma={self.peak_info[i+2]}, Lambda={self.peak_info[i+3]}")

        # Print final fit parameters
        def display_fit_parameters(peak_info, popt, fixed_list=None):
            fit_params_window = tk.Toplevel()
            fit_params_window.title("Final Fit Parameters")
            fit_params_window.geometry("700x200")
            output_text = tk.Text(fit_params_window, wrap=tk.WORD)
            output_text.pack(expand=True, fill=tk.BOTH)

            param_names = ['Amp', 'Mu', 'Sigma', 'Lambda']
            idx = 0
            fixed_param_idx = 0

            # If fixed_list is not provided, create a list of all False values
            if fixed_list is None:
                fixed_list = [False] * len(peak_info)

            for i in range(0, len(peak_info), 4):
                peak_label = f"Peak {(i // 4) + 1}: "
                for j in range(4):
                    if fixed_list[i + j]:
                        peak_label += f"*{param_names[j]}={fixed_list[fixed_param_idx]}, "
                        fixed_param_idx += 1
                    else:
                        peak_label += f"{param_names[j]}={popt[idx]}, "
                        idx += 1
                output_text.insert(tk.END, peak_label + "\n")


        display_fit_parameters(self.peak_info, popt)


        # Plot the histogram and the fitted EMG on the main plot (ax1) with the calibrated x-axis
        plt.hist(self.get_dataset(), bins=num_bins)
        plt.rcParams['figure.figsize'] = [10, 10]
        plt.title(f'Energy Spectrum', fontdict = {'fontsize' : 20})
        plt.xlabel(f'Energy (MeV)', fontdict = {'fontsize' : 20})  # Update the x-axis label
        plt.ylabel(f'Counts',fontdict = {'fontsize' : 20})
        plt.plot(filtered_x_hist, fitted_emg, linestyle='--', label='Multi EMG Fit')  # Use the calibrated x-axis values

        # Convert mu integrated charge to energy
        mu_values = popt[1::4]

        # Add peak labels
        for idx, mu_value in enumerate(mu_values):
            y_value = fitted_emg[np.argmin(np.abs(filtered_x_hist - mu_value))]
            plt.annotate(f"< {mu_value:.2f} MeV | Peak {idx+1}", (mu_value, y_value),textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='black', rotation=90)

        plt.ylim(0, 1.55 * np.max(fitted_emg))
        plt.show()

    
    def multi_fit_from_params(self):
        pass #TODO