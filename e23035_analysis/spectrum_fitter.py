import ROOT

from e23035_analysis import fitting_tools

class spectrum_fitter:
    '''
    Class for easilty fitting 1D spectra and assigning peaks.
    '''
    def __init__(self, spectrum:ROOT.TH1D, peak_model:str):
        '''
        peak_model: gaus for gaussian, or emg for exponentially modified gaussian
        '''
        self.spectrum = spectrum
        #peak location guesses should contain a list of (peak location guess, lower window, upper window)
        self.peaks_to_fit = []
        self.peak_model = peak_model

         #list of dictionaries where each entry corresponds to a peak that was fit
        self.fit_results = []

    def find_peaks(self, reset_peaks=True, sigma=2.0, threshold=0.05, max_peaks=1000, window_width=None):
        '''
        Use SNIP algorithm built into TSpectrum class to build a set of peak location guesses
        and populate peaks_to_fit.
        
        sigma: expected width of the peaks (in bins).
        threshold: peaks with an amplitude less than threshold * highest_peak are ignored (0.0 to 1.0).
        max_peaks: absolute maximum number of peaks to allow.
        window_width: +/- range around the peak for the fit window. If None, defaults to 5 * sigma * bin_width.
        '''
        if reset_peaks:
            self.peaks_to_fit = []
            
        spec = ROOT.TSpectrum(max_peaks) 
        n_found = spec.Search(self.spectrum, sigma, "new", threshold) 
        
        x_peaks = spec.GetPositionX()
        found_locs = [x_peaks[i] for i in range(n_found)]
        found_locs.sort()
        
        bin_width = self.spectrum.GetBinWidth(1)
        actual_width = window_width if window_width is not None else (5.0 * sigma * bin_width)
        
        for loc in found_locs:
            self.peaks_to_fit.append((loc, loc - actual_width, loc + actual_width))
            
        return self.peaks_to_fit

    def fit_peaks(self):
        '''
        Fit each peak from peak_loc_guesses, and store the results
        '''
        original_batch_state = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(True)
        
        for loc_guess, window_start, window_end in self.peaks_to_fit:
            # Define a fit window around the guess (e.g., +/- 2% of energy)
            fit_range = (window_start, window_end)
            location_wiggle = (window_end - window_start) / 2.0

            if self.peak_model.lower() == 'gaus':
                res = fitting_tools.fit_gaussian_peak(
                    self.spectrum, 'gamma_adc', loc_guess, fit_range, param_bounds={'mu': (loc_guess - location_wiggle, loc_guess + location_wiggle)}
                )
            elif self.peak_model.lower() == 'emg':
                res = fitting_tools.fit_emg_peak(
                    self.spectrum, 'gamma_adc', loc_guess, fit_range, param_bounds={'mu': (loc_guess - location_wiggle, loc_guess + location_wiggle)}
                )
            else:
                raise ValueError(f"Unknown peak model: {self.peak_model}")

            # fitting_tools returns (fit_res, background, peak_func, rp, canvas, spectrum_to_plot, f_to_fit, h_fit)
            res_dict = {
                'fit_res': res[0],
                'background_func': res[1],
                'peak_func': res[2],
                'ratio_plot': res[3],
                'canvas': res[4],
                'spectrum_to_plot': res[5],
                'f_to_fit': res[6],
                'h_fit': res[7]
            }
            self.fit_results.append(res_dict)
            
        ROOT.gROOT.SetBatch(original_batch_state)

    def show_fit_results(self, peak_index):
        if 0 <= peak_index < len(self.fit_results):
            canvas = self.fit_results[peak_index]['canvas']
            canvas.Draw()
        else:
            print(f"Invalid peak index: {peak_index}")

    