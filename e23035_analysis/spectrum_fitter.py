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

    def find_peaks(self, reset_peaks=True, expected_peak_width=1.5, window_width=None, required_significance=3.0, plot=True):
        '''
        Finds peaks by identifying all local maxima and filtering them based on statistical significance.
        This method avoids a global threshold, making it suitable for spectra with peaks of varying amplitudes.
        
        expected_peak_width: expected width of the peaks (in x-axis units).
        window_width: +/- range (in x-axis units) around the peak for the fit window. If None, defaults to 5 * expected_peak_width.
        required_significance: The minimum significance (in sigma) for a local maximum to be considered a peak.
        plot: If True, draws a TPolyMarker (stars) on the spectrum at the found peak locations.
        '''
        if reset_peaks:
            self.peaks_to_fit = []
            
        # --- 1. Generate the SNIP Background for Significance Testing ---
        bin_width = self.spectrum.GetBinWidth(1)
        expected_peak_width_bins = max(1, int(expected_peak_width / bin_width))
            
        spec_bg = ROOT.TSpectrum()
        bg_hist = spec_bg.Background(self.spectrum, int(2 * expected_peak_width_bins), "goff")
        bg_hist.SetDirectory(0)
        self.bg_hist = bg_hist # Keep alive in memory

        # --- NEW 2: Strict Local Maxima Search on the RAW Spectrum ---
        candidate_locs = []
        n_bins = self.spectrum.GetNbinsX()
        
        # Define how many bins left/right we check. Half the expected width is usually a safe net.
        search_w = max(1, int(expected_peak_width_bins / 2)) 

        for i in range(search_w + 1, n_bins - search_w):
            val = self.spectrum.GetBinContent(i)
            
            # Fast skip for empty/dead noise regions
            if val < 5: 
                continue 

            # Strict local maximum check: Must be >= all neighbors in the window
            is_max = True
            for j in range(i - search_w, i + search_w + 1):
                if i == j: 
                    continue
                if self.spectrum.GetBinContent(j) > val:
                    is_max = False
                    break

            if is_max:
                candidate_locs.append(self.spectrum.GetXaxis().GetBinCenter(i))

        # --- 3. Filter candidates by significance ---
        valid_peaks = []
        for loc in candidate_locs:
            bin_min = self.spectrum.GetXaxis().FindBin(loc - 2.0 * expected_peak_width)
            bin_max = self.spectrum.GetXaxis().FindBin(loc + 2.0 * expected_peak_width)
            
            signal_sum = 0.0
            err_sq_sum = 0.0
            for i in range(bin_min, bin_max + 1):
                signal_sum += (self.spectrum.GetBinContent(i) - bg_hist.GetBinContent(i))
                err_sq_sum += self.spectrum.GetBinError(i)**2
            
            sig = signal_sum / (err_sq_sum**0.5) if err_sq_sum > 0 else 0.0
            
            if sig >= required_significance:
                y_val = self.spectrum.GetBinContent(self.spectrum.FindBin(loc))
                valid_peaks.append((loc, y_val))
        
        found_peaks = valid_peaks
            
        # --- 4. Update the TPolyMarker for visualization ---
        pm = self.spectrum.GetListOfFunctions().FindObject("TPolyMarker")
        if pm:
            self.spectrum.GetListOfFunctions().Remove(pm)
            
        if plot and len(found_peaks) > 0:
            if not hasattr(self, '_peak_canvas') or not self._peak_canvas:
                ROOT.gROOT.SetBatch(False) 
                self._peak_canvas = ROOT.TCanvas(f"c_peaks_{id(self)}", f"Peak Search: {self.spectrum.GetName()}", 1000, 600)
            
            self._peak_canvas.cd()
            self.spectrum.SetStats(0)
            self.spectrum.Draw("HIST") 
            
            new_pm = ROOT.TPolyMarker(len(found_peaks))
            new_pm.SetMarkerStyle(23)
            new_pm.SetMarkerColor(ROOT.kRed)
            new_pm.SetMarkerSize(1.3)
            for i, (loc, y_val) in enumerate(found_peaks):
                new_pm.SetPoint(i, loc, y_val)
                
            self.spectrum.GetListOfFunctions().Add(new_pm)
            self._poly_marker = new_pm 
            new_pm.Draw()
            self._peak_canvas.Update()
        
        # --- 5. Prepare fit windows for the valid peaks ---
        found_locs = [p[0] for p in found_peaks]
        expected_width_x = window_width if window_width is not None else (5.0 * expected_peak_width)
        
        for loc in found_locs:
            self.peaks_to_fit.append((loc, loc - expected_width_x, loc + expected_width_x))
            
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
            orig_canvas = self.fit_results[peak_index]['canvas']
            
            # The original canvas was created in ROOT Batch mode, so it lacks a GUI window.
            # We create a new canvas and clone the contents of the batch canvas into it.
            new_canvas = ROOT.TCanvas(f"c_show_{peak_index}_{id(self)}", orig_canvas.GetTitle(), 800, 600)
            orig_canvas.DrawClonePad()
            new_canvas.Update()
            
            # Keep a reference to prevent garbage collection from immediately closing the window
            self.fit_results[peak_index]['display_canvas'] = new_canvas
        else:
            print(f"Invalid peak index: {peak_index}")
    