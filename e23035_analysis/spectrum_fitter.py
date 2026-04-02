import ROOT
import numpy as np
from tqdm import tqdm

from e23035_analysis import fitting_tools

class spectrum_fitter:
    '''
    Class for easilty fitting 1D spectra and assigning peaks.
    '''
    def __init__(self, spectrum:ROOT.TH1D, peak_model:str):
        '''
        peak_model: gaus for gaussian, or emg for exponentially modified gaussian, bg_shift_gaus for gaussian with different background to the left and right
        '''
        self.spectrum = spectrum
        #peak location guesses should contain a list of (peak location guess, lower window, upper window)
        self.peaks_to_fit = []
        self.peak_model = peak_model

         #list of dictionaries where each entry corresponds to a peak that was fit
        self.fit_results = []

        self.param_bound_functions = {} #parameter bounds as a function of energy. May be used to fix sigma, etc

    def get_peak_index(self, peak_energy, etol=1):
        '''
        Gets index of peak with specified peak energy, if a peak exists within etol of this value
        Raises value error if more than one peak exists, or if no peaks exist.
        '''
        peak_locs = np.array(self.peaks_to_fit)[:,0]
        i = np.where(np.abs(peak_locs - peak_energy) < etol)[0]
        if len(i) == 0:
            raise ValueError('no peaks found')
        elif len(i) > 1:
            raise ValueError('more than one peak found')
        else:
            return i[0]

    def find_peaks(self, reset_peaks=True, expected_peak_width=1.5, window_width=None, init_sig=3.0, fit_sig=0):
        '''
        Finds peaks by identifying all local maxima and filtering them based on statistical significance.
        
        expected_peak_width: expected width of the peaks (in x-axis units).
        window_width: +/- range (in x-axis units) around the peak for the fit window. If None, defaults to 5 * expected_peak_width.
        init_sig: The minimum significance (in sigma) for a local maximum to be considered a peak candidate. 
        fit_sig: If non-zero, each peak candidate will be fit, and included only if the amplitude/(amplitude uncertainty) > fit_sig
        '''
        if reset_peaks:
            self.peaks_to_fit = []
            
        # --- 1. Generate the SNIP Background for Significance Testing ---
        bin_width = self.spectrum.GetBinWidth(1)
        expected_peak_width_bins = max(1, int(expected_peak_width / bin_width))
            
        spec_bg = ROOT.TSpectrum()
        bg_hist = spec_bg.Background(self.spectrum, 20, "goff")
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
            bin_min = self.spectrum.GetXaxis().FindBin(loc - 1.5 * expected_peak_width)
            bin_max = self.spectrum.GetXaxis().FindBin(loc + 1.5 * expected_peak_width)
            
            signal_sum = 0.0
            err_sq_sum = 0.0
            for j in range(bin_min, bin_max + 1):
                signal_sum += (self.spectrum.GetBinContent(j) - bg_hist.GetBinContent(j))
                err_sq_sum += self.spectrum.GetBinError(j)**2
            
            sig = signal_sum / (err_sq_sum**0.5) if err_sq_sum > 0 else 0.0
            
            if sig >= init_sig:
                y_val = self.spectrum.GetBinContent(self.spectrum.FindBin(loc))
                valid_peaks.append((loc, y_val))
        
        found_peaks = valid_peaks
            
        # --- 4. Prepare fit windows for the valid peaks ---
        found_locs = [p[0] for p in found_peaks]
        expected_width_x = window_width if window_width is not None else (5.0 * expected_peak_width)
        
        for loc in found_locs:
            self.peaks_to_fit.append((loc, loc - expected_width_x, loc + expected_width_x))

        if fit_sig > 0:
            self.fit_peaks()
            
            # Filter peaks based on significance after fitting
            filtered_peaks_to_fit = []
            filtered_fit_results = []
            for i, res_dict in enumerate(self.fit_results):
                amplitude_val, amplitude_err_val = self.get_fit_param_for_peak(i, 'amplitude')
                significance = amplitude_val / amplitude_err_val if amplitude_err_val > 0 else 0
                if significance >= fit_sig:
                    filtered_peaks_to_fit.append(self.peaks_to_fit[i])
                    filtered_fit_results.append(res_dict)
            self.peaks_to_fit = filtered_peaks_to_fit
            self.fit_results = filtered_fit_results

            
        return self.peaks_to_fit

    def show_peak_locations(self, plot_background=False):
        # Remove any existing TPolyMarker from the original spectrum
        # This is important because TSpectrum.Search() adds it by default.
        pm_orig = self.spectrum.GetListOfFunctions().FindObject("TPolyMarker")
        if pm_orig:
            self.spectrum.GetListOfFunctions().Remove(pm_orig)
            
        if len(self.peaks_to_fit) > 0:
            if not hasattr(self, '_peak_canvas') or not self._peak_canvas:
                ROOT.gROOT.SetBatch(False) 
                self._peak_canvas = ROOT.TCanvas(f"c_peaks_{id(self)}", f"Peak Search: {self.spectrum.GetName()}", 1000, 600)
            
            # Clone the spectrum to avoid adding markers to the original histogram
            display_spectrum = self.spectrum.Clone(f"{self.spectrum.GetName()}_display_{id(self)}")
            display_spectrum.SetDirectory(0) # Detach from ROOT's memory management
            self._display_spectrum = display_spectrum # Keep a reference to prevent GC

            self._peak_canvas.cd()
            self._display_spectrum.SetStats(0)
            self._display_spectrum.Draw("HIST") 

            if plot_background and hasattr(self, 'bg_hist') and self.bg_hist:
                self.bg_hist.SetLineColor(ROOT.kGreen+2)
                self.bg_hist.SetLineStyle(2) # Dashed line
                self.bg_hist.SetLineWidth(2)
                self.bg_hist.Draw("HIST SAME")
            
            new_pm = ROOT.TPolyMarker(len(self.peaks_to_fit)) # Create a new TPolyMarker
            new_pm.SetMarkerStyle(23)
            new_pm.SetMarkerColor(ROOT.kRed)
            new_pm.SetMarkerSize(1.3)
            for i, peak_info in enumerate(self.peaks_to_fit):
                loc = peak_info[0]
                # Get Y-value from the cloned spectrum
                y_val = self._display_spectrum.GetBinContent(self._display_spectrum.FindBin(loc))
                new_pm.SetPoint(i, loc, y_val)
                
            self._display_spectrum.GetListOfFunctions().Add(new_pm) # Add to the cloned spectrum
            self._poly_marker = new_pm 
            new_pm.Draw("SAME")
            self._peak_canvas.SetLogy(1)
            self._peak_canvas.Update()

    def fit_peaks(self):
        '''
        Fit each peak from peak_loc_guesses, and store the results
        '''
        self.fit_results = []
        original_batch_state = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(True)
        
        for loc_guess, window_start, window_end in tqdm(self.peaks_to_fit):
            # Define a fit window around the guess (e.g., +/- 2% of energy)
            fit_range = (window_start, window_end)
            location_wiggle = (window_end - window_start) / 2.0

            param_bounds = {}
            if 'mu' not in self.param_bound_functions:
                param_bounds['mu'] = (loc_guess - location_wiggle, loc_guess + location_wiggle)
            for p in self.param_bound_functions:
                param_bounds[p] = self.param_bound_functions[p](loc_guess)

            if self.peak_model.lower() == 'gaus':
                res = fitting_tools.fit_gaussian_peak(
                    self.spectrum, 'gamma_adc', loc_guess, fit_range, param_bounds=param_bounds
                )
            elif self.peak_model.lower() == 'emg':
                res = fitting_tools.fit_emg_peak(
                    self.spectrum, 'gamma_adc', loc_guess, fit_range, param_bounds=param_bounds
                )
            elif self.peak_model.lower() == 'bg_shift_gaus':
                res = fitting_tools.fit_gaussian_w_bg_shift(self.spectrum, loc_guess, fit_range, 
                                    param_bounds=param_bounds)
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

    def get_fit_param(self, param_name):
        '''
        Returns (param value array, param error array) containing values for all peaks
        '''
        vals = []
        errs = []
        for res in self.fit_results:
            fit_res = res['fit_res']
            f_to_fit = res['f_to_fit']
            found = False
            for i in range(f_to_fit.GetNpar()):
                if f_to_fit.GetParName(i) == param_name:
                    vals.append(fit_res.Parameter(i))
                    errs.append(fit_res.ParError(i))
                    found = True
                    break
            if not found:
                raise ValueError('invalid parameter: %s'%param_name)
        return np.array(vals), np.array(errs)

    def get_fit_param_for_peak(self, peak_index, param_name):
        '''
        Returns (param value, param error) for a specific peak.
        '''
        if not (0 <= peak_index < len(self.fit_results)):
            raise IndexError(f"Peak index {peak_index} is out of bounds.")

        res = self.fit_results[peak_index]
        fit_res = res['fit_res']
        f_to_fit = res['f_to_fit']

        for i in range(f_to_fit.GetNpar()):
            if f_to_fit.GetParName(i) == param_name:
                return fit_res.Parameter(i), fit_res.ParError(i)
        
        raise ValueError(f"Invalid parameter: {param_name} for peak {peak_index}")

    def get_fit_probs(self):
        to_return = []
        for res in self.fit_results:
            to_return.append(res['fit_res'].Prob())
        return to_return

    def show_fit_results(self, peak_index):
        if 0 <= peak_index < len(self.fit_results):
            orig_canvas = self.fit_results[peak_index]['canvas']
            
            # 1. Create new canvas and clone the visual state
            new_canvas = ROOT.TCanvas(f"c_show_{peak_index}_{id(self)}", orig_canvas.GetTitle(), 800, 600)
            orig_canvas.DrawClonePad()
            new_canvas.Update()

            # Retrieve fit results and function
            fit_res = self.fit_results[peak_index]['fit_res']
            f_to_fit = self.fit_results[peak_index]['f_to_fit']

            # 2. Find the upper pad of the cloned TRatioPlot
            # TRatioPlot creates custom TPads. The upper pad is the first TPad primitive.
            new_canvas.cd()
            for prim in new_canvas.GetListOfPrimitives():
                if prim.InheritsFrom("TPad"):
                    prim.cd()
                    break

            # 3. Create a custom stats box
            # Coordinates are Normalized Device Coordinates (NDC)
            stats_box = ROOT.TPaveText(0.65, 0.45, 0.88, 0.88, "NDC")
            stats_box.SetFillColor(ROOT.kWhite)
            stats_box.SetBorderSize(1)
            stats_box.SetTextAlign(12) # Left-align the text

            # Add Goodness of Fit info
            prob = fit_res.Prob()
            chi2_ndf = fit_res.Chi2() / fit_res.Ndf() if fit_res.Ndf() > 0 else 0
            stats_box.AddText(f"P-value: {prob:.4g}")
            stats_box.AddText(f"#chi^{{2}}/ndf: {chi2_ndf:.2f}")
            stats_box.AddLine(0, 0, 0, 0)

            # Dynamically add all parameters
            for i in range(fit_res.NPar()):
                p_name = f_to_fit.GetParName(i)
                p_val = fit_res.Parameter(i)
                p_err = fit_res.ParError(i)
                
                # Format ROOT LaTeX strings
                if p_name == "mu": p_name = "#mu"
                elif p_name == "sigma": p_name = "#sigma"
                elif p_name == "tau": p_name = "#tau"
                
                stats_box.AddText(f"{p_name}: {p_val:.4g} #pm {p_err:.4g}")

            # 4. Draw and Update
            stats_box.Draw("SAME")
            new_canvas.Update()

            # 5. Keep references to prevent Python from deleting the GUI objects!
            self.fit_results[peak_index]['display_canvas'] = new_canvas
            self.fit_results[peak_index]['stats_box'] = stats_box 
            
        else:
            print(f"Invalid peak index: {peak_index}")