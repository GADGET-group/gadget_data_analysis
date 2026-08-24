import os
import csv
import re

import dill
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
        peak_model: gaus for gaussian, or emg for exponentially modified gaussian, bg_shift_gaus for gaussian with different background to the left and right,
        or bg_shift_emg
        '''
        self.spectrum = spectrum
        #peak location guesses should contain a list of (peak location guess, lower window, upper window)
        #The location guess may contain an list location guesses if more than one peak should be fit
        self.peaks_to_fit = []
        self.peak_model = peak_model

         #list of dictionaries where each entry corresponds to a peak that was fit
        self.fit_results = []
        
        #parameter bounds as a function of energy. May be used to fix sigma, etc
        #These functions will be evaluated at loc_guess[0] if loc guess is a list of locations
        self.param_bound_functions = {}
        self.fit_options = 'LS0QEI'#defaults to log likelihood. Should change if fitting a bg subtracted spectrum.
        self.location_wiggle=3 #bounds +/- to apply to location guesses
        self.shared_sigma=True #currently only implemented for bg_shift_guass. Will fill out param bounds for each peak based on guess.
        self.max_implicit_cores = 100
        self.parameterizations = {}

    def add_peaks(self, peak_locations, window_size, sep_factor=1.25):
        '''
        peak_locations: list of peak locations
        window_size: function of energy, used to determine fit window and which peaks overlap with each other

        Add peaks located a peaks_to_fit. Peaks less than sep_factor*window_size(peak_location) apart  will be
        fit in a combined window. The fit window will run from first_peak_location-window_size to
        last_peak_location+window_size.
        '''
        peak_locations = np.sort(peak_locations)
        i = 0
        while i < len(peak_locations):
            group_locations = [peak_locations[i]]
            while i < len(peak_locations) - 1:
                dE = peak_locations[i+1] - peak_locations[i]
                if dE < sep_factor*max(window_size(peak_locations[i+1]), window_size(peak_locations[i])):
                    group_locations.append(peak_locations[i+1])
                    i += 1
                else:
                    break
            self.peaks_to_fit.append((group_locations, group_locations[0] - window_size(group_locations[0]),
                                      group_locations[-1] + window_size(group_locations[-1])))
            i += 1


    def save(self, filepath):
        '''
        Save fit results, canvases, histograms, and python state to a single {file_path}.root file.
        Save fit peak locations and parameters with uncertainties to {filepath}.csv
        '''        
        if not filepath.endswith('.root'):
            filepath += '.root'

        csv_filepath = filepath[:-5] + '.csv'

        print(f"Saving spectrum fitter state to {filepath} and {csv_filepath}...")
        
        with open(csv_filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # 1. Identify all unique free parameter base names across all fits
            free_base_params = set()
            for res in self.fit_results:
                if res is None: continue
                fit_res = res.get('fit_res')
                f_to_fit = res.get('f_to_fit')
                if not fit_res or not f_to_fit: continue
                
                for j in range(f_to_fit.GetNpar()):
                    # Check if the parameter is actively free in the fit
                    param_free = True
                    try:
                        if hasattr(fit_res, "Get") and fit_res.Get():
                            param_free = not fit_res.Get().IsParameterFixed(j)
                        elif hasattr(fit_res, "IsParameterFixed"):
                            param_free = not fit_res.IsParameterFixed(j)
                        else:
                            param_free = (fit_res.ParError(j) != 0.0)
                    except Exception:
                        param_free = (fit_res.ParError(j) != 0.0)
                    
                    if param_free:
                        name = f_to_fit.GetParName(j)
                        match = re.match(r'^(.*)_(\d+)$', name)
                        if match:
                            free_base_params.add(match.group(1))
                        else:
                            free_base_params.add(name)
            
            sorted_params = sorted(list(free_base_params))
            priorities = {'mu': 1, 'amplitude': 2, 'sigma': 3, 'bg_const': 4}
            sorted_params.sort(key=lambda x: (priorities.get(x, 100), x))
            
            header = ['fit_index', 'loc_guess', 'p_value']
            for p in sorted_params:
                header.extend([f'{p}_val', f'{p}_err'])
            csvwriter.writerow(header)

            for i, res in enumerate(self.fit_results):
                if res is None: continue
                
                loc_guesses = self.peaks_to_fit[i][0]
                if not isinstance(loc_guesses, (list, tuple, np.ndarray)):
                    loc_guesses = [loc_guesses]
                
                fit_res = res.get('fit_res')
                f_to_fit = res.get('f_to_fit')
                if not fit_res or not f_to_fit: continue
                
                p_value = fit_res.Prob()
                
                for k, loc in enumerate(loc_guesses):
                    row_dict = {}
                    for j in range(f_to_fit.GetNpar()):
                        param_free = True
                        try:
                            if hasattr(fit_res, "Get") and fit_res.Get():
                                param_free = not fit_res.Get().IsParameterFixed(j)
                            elif hasattr(fit_res, "IsParameterFixed"):
                                param_free = not fit_res.IsParameterFixed(j)
                            else:
                                param_free = (fit_res.ParError(j) != 0.0)
                        except Exception:
                            param_free = (fit_res.ParError(j) != 0.0)
                            
                        if param_free:
                            name = f_to_fit.GetParName(j)
                            match = re.match(r'^(.*)_(\d+)$', name)
                            if match:
                                base_name = match.group(1)
                                param_k = int(match.group(2))
                                if param_k == k:
                                    row_dict[base_name + '_val'] = fit_res.Parameter(j)
                                    row_dict[base_name + '_err'] = fit_res.ParError(j)
                            else:
                                base_name = name
                                row_dict[base_name + '_val'] = fit_res.Parameter(j)
                                row_dict[base_name + '_err'] = fit_res.ParError(j)
                                
                    row = [i, loc, p_value]
                    for p in sorted_params:
                        row.append(row_dict.get(p + '_val', ''))
                        row.append(row_dict.get(p + '_err', ''))
                    csvwriter.writerow(row)

        f = ROOT.TFile(filepath, "RECREATE")

        # 1. Write the main spectrum
        self.spectrum.Write("main_spectrum")

        # 2. Package standard Python types into a state dictionary
        python_state = {
            'peak_model': self.peak_model,
            'peaks_to_fit': self.peaks_to_fit,
            'fit_options': self.fit_options,
            'num_fit_results': len(self.fit_results),
            'parameterizations': self.parameterizations
        }

        # We extract and store just the source code of the lambda functions 
        # to avoid serializing the global namespace (which causes crashes).
        import inspect
        source_dict = {}
        for k, v in self.param_bound_functions.items():
            try:
                source_dict[k] = inspect.getsource(v).strip()
            except Exception:
                source_dict[k] = str(v)
        python_state['param_bound_functions'] = source_dict

        # 3. Iterate through fit results and save ROOT objects
        for i, res in enumerate(self.fit_results):
            if res is None: continue # Skip if parallel fitting failed for an index

            
            # TFitResultPtr cannot be written directly. 
            # We must call .Get() to extract the underlying C++ TFitResult object.
            if res.get('fit_res') and res['fit_res'].Get():
                res['fit_res'].Get().Write(f"peak_{i}_fit_res")
                
            if res.get('component_peak_funcs'):
                for j, comp_func in enumerate(res['component_peak_funcs']):
                    comp_func.Write(f"peak_{i}_component_{j}")

            if res.get('background_func'): res['background_func'].Write(f"peak_{i}_background_func")
            if res.get('peak_func'): res['peak_func'].Write(f"peak_{i}_peak_func")
            if res.get('ratio_plot'): res['ratio_plot'].Write(f"peak_{i}_ratio_plot")
            if res.get('canvas'): res['canvas'].Write(f"peak_{i}_canvas")
            if res.get('spectrum_to_plot'): res['spectrum_to_plot'].Write(f"peak_{i}_spectrum_to_plot")
            if res.get('f_to_fit'): res['f_to_fit'].Write(f"peak_{i}_f_to_fit")
            if res.get('h_fit'): res['h_fit'].Write(f"peak_{i}_h_fit")

        # 4. Serialize the Python state into a Hex String and save as TObjString
        # Hex encoding prevents ROOT from mangling null bytes during string conversion
        pickled_hex_string = dill.dumps(python_state).hex()
        obj_string = ROOT.TObjString(pickled_hex_string)
        obj_string.Write("python_state")

        f.Close()
        print("Save complete.")

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

    def get_fit_param_for_peak(self, peak_index, param_name):
        """
        Gets a fit parameter for a specific peak fit, identified by its index.
        This is for single-peak fits generated by find_peaks.
        """
        if not (0 <= peak_index < len(self.fit_results)):
            raise IndexError(f"peak_index {peak_index} is out of bounds.")
        
        res_dict = self.fit_results[peak_index]
        if res_dict is None or 'fit_res' not in res_dict or 'f_to_fit' not in res_dict:
            raise ValueError(f"No valid fit result for peak_index {peak_index}")

        fit_res = res_dict['fit_res']
        f_to_fit = res_dict['f_to_fit']

        for i in range(f_to_fit.GetNpar()):
            if f_to_fit.GetParName(i) == param_name:
                return fit_res.Parameter(i), fit_res.ParError(i)
                
        raise ValueError(f"Parameter '{param_name}' not found in fit for peak_index {peak_index}")


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
        original_mt_state = ROOT.IsImplicitMTEnabled()
        if not original_mt_state and self.peak_model.lower() not in ['emg', 'bg_shift_emg', 'bg_shift_nemg']:
            ROOT.EnableImplicitMT(self.max_implicit_cores)
        self.fit_results = []
        original_batch_state = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(True)
        
        for loc_guess, window_start, window_end in tqdm(self.peaks_to_fit):
            try:
                _ = iter(loc_guess)
            except TypeError as te:
                loc_guess = [loc_guess]
            print(f'fitting peak/s at:{loc_guess}')

            # Define a fit window around the guess (e.g., +/- 2% of energy)
            fit_range = (window_start, window_end)
            location_wiggle = self.location_wiggle#(window_end - window_start) / 2.0

            param_bounds = {}
            
            if len(loc_guess) == 1:
                if 'mu' in self.param_bound_functions:
                    param_bounds['mu'] = self.param_bound_functions['mu'](loc_guess[0])
                elif 'mu' not in self.param_bound_functions:
                    param_bounds['mu'] = (loc_guess[0] - location_wiggle, loc_guess[0] + location_wiggle)
            else:
                for i, loc in enumerate(loc_guess):
                    if 'mu' in self.param_bound_functions:
                        param_bounds[f'mu_{i}'] = self.param_bound_functions['mu'](loc)
                    elif f'mu_{i}' not in self.param_bound_functions:
                        param_bounds[f'mu_{i}'] = (loc - location_wiggle, loc + location_wiggle)
                        
            for p in self.param_bound_functions:
                if p == 'mu':
                    continue
                param_bounds[p] = self.param_bound_functions[p](loc_guess[0])

            if self.peak_model.lower() == 'gaus':
                res = fitting_tools.fit_gaussian_peak(
                    self.spectrum, 'gamma_adc', loc_guess, fit_range, param_bounds=param_bounds, fit_options=self.fit_options,
                    parameterizations=self.parameterizations
                )
            elif self.peak_model.lower() == 'emg':
                res = fitting_tools.fit_emg_peak(
                    self.spectrum, 'gamma_adc', loc_guess, fit_range, param_bounds=param_bounds, fit_options=self.fit_options,
                    parameterizations=self.parameterizations
                )
            elif self.peak_model.lower() == 'bg_shift_gaus':
                if not self.shared_sigma and len(loc_guess)>1 and 'sigma' in self.param_bound_functions:
                    del param_bounds['sigma']
                    for i, loc in enumerate(loc_guess):
                        param_bounds[f'sigma_{i}'] = self.param_bound_functions['sigma'](loc)
                
                res = fitting_tools.fit_gaussian_w_bg_shift(self.spectrum, loc_guess, fit_range, 
                                    param_bounds=param_bounds, fit_options=self.fit_options, shared_sigma=self.shared_sigma,
                                    parameterizations=self.parameterizations)
            elif self.peak_model.lower() == 'bg_shift_emg':
                res = fitting_tools.fit_emg_w_bg_shift(self.spectrum, loc_guess, fit_range, 
                                    param_bounds=param_bounds, fit_options=self.fit_options,
                                    parameterizations=self.parameterizations)
            elif self.peak_model.lower() == 'bg_shift_ngaus':
                res = fitting_tools.fit_ngaussian_w_bg_shift(self.spectrum, loc_guess, fit_range, self.ngaus,
                                    param_bounds=param_bounds, fit_options=self.fit_options)
            elif self.peak_model.lower() == 'bg_shift_voigt':
                res = fitting_tools.fit_voigt_w_bg_shift(self.spectrum, loc_guess, fit_range,
                                    param_bounds=param_bounds, fit_options=self.fit_options)
            elif self.peak_model.lower() == 'bg_shift_nemg':
                res = fitting_tools.fit_nemg_w_bg_shift(self.spectrum, loc_guess, fit_range, self.nemg,
                                    param_bounds=param_bounds, fit_options=self.fit_options)
            else:
                raise ValueError(f"Unknown peak model: {self.peak_model}")

            # fitting_tools returns (fit_res, background, peak_func, component_peak_funcs, rp, canvas, spectrum_to_plot, f_to_fit, h_fit)
            res_dict = {
                'fit_res': res[0],
                'background_func': res[1],
                'peak_func': res[2],
                'component_peak_funcs': res[3],
                'ratio_plot': res[4],
                'canvas': res[5],
                'spectrum_to_plot': res[6],
                'f_to_fit': res[7],
                'h_fit': res[8]
            }
            self.fit_results.append(res_dict)
            
        ROOT.gROOT.SetBatch(original_batch_state)
        if not original_mt_state:
            ROOT.DisableImplicitMT()

    def get_fit_param(self, param_name):
        '''
        Returns (param value array, param error array) containing values for all peaks
        '''
        vals = []
        errs = []
        for res in self.fit_results:
            if res is None or 'fit_res' not in res or 'f_to_fit' not in res:
                continue
            fit_res = res['fit_res']
            f_to_fit = res['f_to_fit']
            for i in range(f_to_fit.GetNpar()):
                par_name_i = f_to_fit.GetParName(i)
                # Check if the parameter name is exactly the base name OR starts with the base name + '_'
                if par_name_i == param_name or par_name_i.startswith(param_name + '_'):
                    vals.append(fit_res.Parameter(i))
                    errs.append(fit_res.ParError(i))

        if not vals:
            raise ValueError('invalid parameter: %s' % param_name)
        return np.array(vals), np.array(errs)

    def get_param_for_guess(self, param_base_name, guess_value):
        """
        Finds the fitted value of a parameter for a specific initial peak guess.
        Returns (value, error) or (None, None) if not found.
        """
        for i, (loc_guesses, _, _) in enumerate(self.peaks_to_fit):
            # Ensure loc_guesses is a list
            if not isinstance(loc_guesses, (list, tuple, np.ndarray)):
                loc_guesses = [loc_guesses]
            
            try:
                # Find the index of the guess value in the list of guesses for this fit
                k = list(loc_guesses).index(guess_value)
            except ValueError:
                # The guess value is not in this group of peaks
                continue

            # We found the fit, it's at index i, and it's the k-th peak in that fit.
            res_dict = self.fit_results[i]
            if res_dict is None or 'fit_res' not in res_dict or 'f_to_fit' not in res_dict:
                continue
            
            fit_res = res_dict['fit_res']
            f_to_fit = res_dict['f_to_fit']

            # Determine the parameter name to search for
            if len(loc_guesses) == 1:
                # For single-peak fits, the parameter name is just the base name
                param_name_to_find = param_base_name
            else:
                # For multi-peak fits, it's base_name + '_' + index
                param_name_to_find = f'{param_base_name}_{k}'

            # Find the parameter in the function and return its value and error
            for j in range(f_to_fit.GetNpar()):
                if f_to_fit.GetParName(j) == param_name_to_find:
                    return fit_res.Parameter(j), fit_res.ParError(j)
        
        return None, None

    def get_fit_probs(self):
        to_return = []
        for res in self.fit_results:
            to_return.append(res['fit_res'].Prob())
        return to_return

    def show_fit_results(self, peak_index, show_fit_params=True, show_components=False):
        ROOT.gROOT.SetBatch(False) 
        if 0 <= peak_index < len(self.fit_results):
            orig_canvas = self.fit_results[peak_index]['canvas']
            
            # Ensure data is black and fit is red for previously saved fits
            spec = self.fit_results[peak_index].get('spectrum_to_plot')
            if spec:
                spec.SetLineColor(ROOT.kBlack)
                spec.SetMarkerColor(ROOT.kBlack)
            h_fit = self.fit_results[peak_index].get('h_fit')
            if h_fit:
                h_fit.SetLineColor(ROOT.kRed)
            f_to_fit = self.fit_results[peak_index].get('f_to_fit')
            if f_to_fit:
                f_to_fit.SetLineColor(ROOT.kRed)
            rp = self.fit_results[peak_index].get('ratio_plot')
            if rp and hasattr(rp, "GetLowerRefGraph") and rp.GetLowerRefGraph():
                rp.GetLowerRefGraph().SetLineColor(ROOT.kBlack)
                rp.GetLowerRefGraph().SetMarkerColor(ROOT.kBlack)
                
            orig_canvas.Modified()
            orig_canvas.Update()

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
            upper_pad = None
            for prim in new_canvas.GetListOfPrimitives():
                if prim.InheritsFrom("TPad"):
                    prim.cd()
                    upper_pad = prim
                    break
            
            if upper_pad:
                upper_pad.cd()
                if show_components:
                    bg_func = self.fit_results[peak_index].get('background_func')
                    if bg_func:
                        bg_func.SetLineColor(ROOT.kRed)
                        bg_func.SetLineStyle(2)
                        bg_func.Draw("SAME")
                        
                    component_funcs = self.fit_results[peak_index].get('component_peak_funcs')
                    if component_funcs:
                        for i, func in enumerate(component_funcs):
                            func.SetLineStyle(3)
                            func.SetLineColor(ROOT.kRed)
                            func.Draw("SAME")
                    else:
                        peak_func = self.fit_results[peak_index].get('peak_func')
                        if peak_func:
                            peak_func.SetLineColor(ROOT.kRed)
                            peak_func.SetLineStyle(3)
                            peak_func.Draw("SAME")

            if show_fit_params:
                if upper_pad:
                    upper_pad.cd()
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
            if show_fit_params:
                self.fit_results[peak_index]['stats_box'] = stats_box 
            
        else:
            print(f"Invalid peak index: {peak_index}")

def load_spectrum_fitter_from_file(file_path) -> 'spectrum_fitter':
    """
    Load a previously saved spectrum fitter from a .root file.
    """
    print(f"Loading spectrum fitter state from {file_path}...")
    f = ROOT.TFile(file_path, "READ")
    if f.IsZombie():
        raise FileNotFoundError(f"ROOT could not open {file_path}")

    # 1. Extract and decode the Python State
    state_obj = f.Get("python_state")
    if not state_obj:
        raise ValueError(f"Invalid file format: 'python_state' missing in {file_path}")
    
    python_state = dill.loads(bytes.fromhex(state_obj.GetString().Data()))

    is_multi = python_state.get('is_multi', False)
    
    if is_multi:
        num_spectra = python_state.get('num_spectra', 0)
        spectra = []
        for idx in range(num_spectra):
            spec = f.Get(f"spectrum_{idx}")
            if not spec:
                raise ValueError(f"Spectrum {idx} missing from file.")
            spec.SetDirectory(0)
            spectra.append(spec)
            
        fitter = multi_spectrum_fitter(spectra, python_state['peak_model'])
        fitter.shared_sigma = python_state.get('shared_sigma', False)
        fitter.location_wiggle = python_state.get('location_wiggle', 10)
    else:
        # 2. Extract Main Spectrum and detach it from the file
        main_spectrum = f.Get("main_spectrum")
        if not main_spectrum:
            raise ValueError("Main spectrum missing from file.")
        main_spectrum.SetDirectory(0) 
        fitter = spectrum_fitter(main_spectrum, python_state['peak_model'])

    fitter.peaks_to_fit = python_state['peaks_to_fit']
    fitter.fit_options = python_state.get('fit_options', 'LS0QEI')
    
    if python_state.get('param_bound_functions'):
        fitter.param_bound_functions = python_state['param_bound_functions']
    
    if python_state.get('parameterizations'):
        fitter.parameterizations = python_state['parameterizations']

    # 4. Reconstruct Fit Results
    num_fits = python_state.get('num_fit_results', 0)
    
    for i in range(num_fits):
        res_dict = {}
        
        # When we read the fit result back, ROOT returns a raw TFitResult. 
        # We wrap it back into a TFitResultPtr to maintain your API.
        fit_res_obj = f.Get(f"peak_{i}_fit_res")
        if fit_res_obj:
            res_dict['fit_res'] = ROOT.TFitResultPtr(fit_res_obj)
        else:
            res_dict['fit_res'] = ROOT.TFitResultPtr() # Empty pointer fallback

        if is_multi:
            sub_hist_2d = f.Get(f"peak_{i}_sub_hist_2d")
            if sub_hist_2d: sub_hist_2d.SetDirectory(0)
            res_dict['sub_hist_2d'] = sub_hist_2d
            
            f_to_fit_2d = f.Get(f"peak_{i}_f_to_fit_2d")
            res_dict['f_to_fit_2d'] = f_to_fit_2d
            
            h_fit_2d = f.Get(f"peak_{i}_h_fit_2d")
            if h_fit_2d: h_fit_2d.SetDirectory(0)
            res_dict['h_fit_2d'] = h_fit_2d
            
            h_resid_2d = f.Get(f"peak_{i}_h_resid_2d")
            if h_resid_2d: h_resid_2d.SetDirectory(0)
            res_dict['h_resid_2d'] = h_resid_2d
            
            canvas_2d = f.Get(f"peak_{i}_canvas_2d")
            res_dict['canvas_2d'] = canvas_2d
            
            if f'peak_{i}_pm_names' in python_state:
                from e23035_analysis import fitting_tools
                pm = fitting_tools.ParamManager()
                pm.names = python_state[f'peak_{i}_pm_names']
                if f'peak_{i}_bg_func_name' in python_state:
                    pm.bg_func_name = python_state[f'peak_{i}_bg_func_name']
                if f'peak_{i}_peak_func_name' in python_state:
                    pm.peak_func_name = python_state[f'peak_{i}_peak_func_name']
                if f'peak_{i}_cpp_code' in python_state:
                    pm.cpp_code = python_state[f'peak_{i}_cpp_code']
                    ROOT.gInterpreter.Declare(pm.cpp_code)
                res_dict['pm'] = pm
        else:
            res_dict['component_peak_funcs'] = []
            j = 0
            while True:
                comp_func = f.Get(f"peak_{i}_component_{j}")
                if comp_func:
                    res_dict['component_peak_funcs'].append(comp_func)
                    j += 1
                else:
                    break

            # Load GUI / Function Objects
            res_dict['background_func'] = f.Get(f"peak_{i}_background_func")
            res_dict['peak_func'] = f.Get(f"peak_{i}_peak_func")
            res_dict['ratio_plot'] = f.Get(f"peak_{i}_ratio_plot")
            res_dict['canvas'] = f.Get(f"peak_{i}_canvas")
            res_dict['f_to_fit'] = f.Get(f"peak_{i}_f_to_fit")
            
            # Load Histograms (Must detach from the file so they aren't deleted)
            spec_plot = f.Get(f"peak_{i}_spectrum_to_plot")
            if spec_plot: spec_plot.SetDirectory(0)
            res_dict['spectrum_to_plot'] = spec_plot
            
            h_fit = f.Get(f"peak_{i}_h_fit")
            if h_fit: h_fit.SetDirectory(0)
            res_dict['h_fit'] = h_fit

        fitter.fit_results.append(res_dict)

    # CRITICAL: Do not close 'f'! 
    # Canvases and TRatioPlots depend on the open file. We attach it to the fitter
    # so it stays alive exactly as long as the fitter object exists.
    fitter._saved_file_reference = f 
    
    print("Load complete.")
    return fitter

class multi_spectrum_fitter(spectrum_fitter):
    '''
    Class for simultaneously fitting multiple 1D spectra.
    '''
    def __init__(self, spectra:list, peak_model:str):
        if not spectra:
            raise ValueError("Must provide at least one spectrum")
        self.spectra = spectra
        # Initialize the base class with the first spectrum to reuse some base class logic
        super().__init__(spectra[0], peak_model)
        
    def find_peaks(self, reset_peaks=True, expected_peak_width=1.5, window_width=None, init_sig=3.0, fit_sig=0, spectrum_index=0):
        '''
        Finds peaks on a specific 1D spectrum.
        '''
        original_spectrum = self.spectrum
        self.spectrum = self.spectra[spectrum_index]
        res = super().find_peaks(reset_peaks, expected_peak_width, window_width, init_sig, fit_sig)
        self.spectrum = original_spectrum
        return res

    def fit_peaks(self):
        '''
        Fit each peak simultaneously across all spectra using the 2D fitter.
        '''
        original_mt_state = ROOT.IsImplicitMTEnabled()
        if not original_mt_state and self.peak_model.lower() not in ['emg', 'bg_shift_emg', 'bg_shift_nemg']:
            ROOT.EnableImplicitMT(self.max_implicit_cores)
        self.fit_results = []
        original_batch_state = ROOT.gROOT.IsBatch()
        ROOT.gROOT.SetBatch(True)
        
        for loc_guess, window_start, window_end in tqdm(self.peaks_to_fit):
            try:
                _ = iter(loc_guess)
            except TypeError as te:
                loc_guess = [loc_guess]
            print(f'fitting peak/s simultaneously at:{loc_guess}')

            fit_range = (window_start, window_end)
            location_wiggle = self.location_wiggle

            param_bounds = {}
            if len(loc_guess) == 1:
                if 'mu' in self.param_bound_functions:
                    param_bounds['mu'] = self.param_bound_functions['mu'](loc_guess[0])
                elif 'mu' not in self.param_bound_functions:
                    param_bounds['mu'] = (loc_guess[0] - location_wiggle, loc_guess[0] + location_wiggle)
            else:
                for i, loc in enumerate(loc_guess):
                    if 'mu' in self.param_bound_functions:
                        param_bounds[f'mu_{i}'] = self.param_bound_functions['mu'](loc)
                    elif f'mu_{i}' not in self.param_bound_functions:
                        param_bounds[f'mu_{i}'] = (loc - location_wiggle, loc + location_wiggle)
                        
            for p in self.param_bound_functions:
                if p == 'mu':
                    continue
                param_bounds[p] = self.param_bound_functions[p](loc_guess[0])

            if self.peak_model.lower() == 'bg_shift_gaus':
                if not self.shared_sigma and len(loc_guess)>1 and 'sigma' in self.param_bound_functions:
                    del param_bounds['sigma']
                    for i, loc in enumerate(loc_guess):
                        param_bounds[f'sigma_{i}'] = self.param_bound_functions['sigma'](loc)
                
                res = fitting_tools.fit_gaussian_w_bg_shift_2d(self.spectra, loc_guess, fit_range, 
                                    param_bounds=param_bounds, fit_options=self.fit_options, shared_sigma=self.shared_sigma,
                                    parameterizations=self.parameterizations)
            elif self.peak_model.lower() == 'bg_shift_emg':
                res = fitting_tools.fit_emg_w_bg_shift_2d(self.spectra, loc_guess, fit_range, 
                                    param_bounds=param_bounds, fit_options=self.fit_options,
                                    parameterizations=self.parameterizations)
            else:
                raise ValueError(f"Unknown peak model for multi_spectrum_fitter (currently supports bg_shift_gaus, bg_shift_emg): {self.peak_model}")

            # fit_hist2d returns fit_res, canvas, sub_hist, f_to_fit, h_fit, h_resid, pm
            res_dict = {
                'fit_res': res[0],
                'canvas_2d': res[1],
                'sub_hist_2d': res[2],
                'f_to_fit_2d': res[3],
                'h_fit_2d': res[4],
                'h_resid_2d': res[5],
                'pm': res[6]
            }
            self.fit_results.append(res_dict)
            
        ROOT.gROOT.SetBatch(original_batch_state)
        if not original_mt_state:
            ROOT.DisableImplicitMT()

    def show_fit_results(self, peak_index, show_fit_params=True, show_components=False):
        ROOT.gROOT.SetBatch(False) 
        if 0 <= peak_index < len(self.fit_results):
            res = self.fit_results[peak_index]
            fit_res = res['fit_res']
            f_to_fit_2d = res['f_to_fit_2d']
            pm = res['pm']
            
            # The user requested a color coded overlay of all spectra
            new_canvas = ROOT.TCanvas(f"c_show_multi_{peak_index}_{id(self)}", f"Multi-spectrum Fit Result {peak_index}", 1000, 600)
            
            # 1. Main plot with ratio panel below
            pad1 = ROOT.TPad(f"pad1_multi_{id(self)}", "pad1", 0, 0.3, 1, 1.0)
            pad1.SetBottomMargin(0.02)
            pad1.Draw()
            
            new_canvas.cd()
            pad2 = ROOT.TPad(f"pad2_multi_{id(self)}", "pad2", 0, 0.0, 1, 0.3)
            pad2.SetTopMargin(0.02)
            pad2.SetBottomMargin(0.3)
            pad2.Draw()

            pad1.cd()
            
            colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kOrange, ROOT.kMagenta, ROOT.kCyan]
            
            # Collect components for drawing
            e_low = f_to_fit_2d.GetXmin()
            e_high = f_to_fit_2d.GetXmax()
            
            drawn_first = False
            
            # Reconstruct the 1D function for each spectrum
            for j, spec in enumerate(self.spectra):
                color = colors[j % len(colors)]
                
                # Draw the original data spectrum
                spec_clone = spec.Clone(f"spec_clone_{j}_{id(self)}")
                spec_clone.GetXaxis().SetRangeUser(e_low, e_high)
                spec_clone.SetLineColor(color)
                spec_clone.SetMarkerColor(color)
                spec_clone.SetStats(0)
                
                if not drawn_first:
                    spec_clone.Draw("E")
                    drawn_first = True
                else:
                    spec_clone.Draw("E SAME")
                    
                # Create a 1D function bound to this specific y value
                # We must keep a strong reference to the lambda to prevent PyROOT from segfaulting!
                def make_eval(idx):
                    return lambda x, p: f_to_fit_2d.Eval(x[0], idx)
                
                lam = make_eval(j)
                f1d = ROOT.TF1(f"f1d_{j}_{id(self)}", lam, e_low, e_high, 0)
                f1d.SetLineColor(color)
                f1d.SetLineWidth(2)
                f1d.Draw("SAME")
                
                if '1d_lambdas' not in res:
                    res['1d_lambdas'] = []
                res['1d_lambdas'].append(lam)
                
                # We need to keep these references so they don't get garbage collected
                if '1d_funcs' not in res:
                    res['1d_funcs'] = []
                res['1d_funcs'].append((spec_clone, f1d))
                
                if show_components:
                    bg_func_name = getattr(pm, 'bg_func_name', None)
                    if bg_func_name:
                        def make_bg_eval(idx):
                            bg_eval_func = getattr(ROOT, bg_func_name)
                            params = f_to_fit_2d.GetParameters()
                            return lambda x, p: bg_eval_func(np.array([x[0], idx], dtype=np.float64), params)
                            
                        lam_bg = make_bg_eval(j)
                        f1d_bg = ROOT.TF1(f"f1d_bg_{j}_{id(self)}", lam_bg, e_low, e_high, 0)
                        f1d_bg.SetLineColor(color)
                        f1d_bg.SetLineStyle(2)
                        f1d_bg.SetLineWidth(1)
                        f1d_bg.Draw("SAME")
                        res['1d_lambdas'].append(lam_bg)
                        res['1d_funcs'].append((None, f1d_bg))
                        
                    peak_func_name = getattr(pm, 'peak_func_name', None)
                    if peak_func_name:
                        n_peaks = sum(1 for name in pm.names if name.startswith('mu'))
                        for i in range(n_peaks):
                            def make_peak_eval(idx, peak_i):
                                peak_eval_func = getattr(ROOT, peak_func_name)
                                params = f_to_fit_2d.GetParameters()
                                return lambda x, p: peak_eval_func(np.array([x[0], idx, peak_i], dtype=np.float64), params)
                                
                            lam_peak = make_peak_eval(j, i)
                            f1d_peak = ROOT.TF1(f"f1d_peak_{j}_{i}_{id(self)}", lam_peak, e_low, e_high, 0)
                            f1d_peak.SetLineColor(color)
                            f1d_peak.SetLineStyle(3)
                            f1d_peak.SetLineWidth(1)
                            f1d_peak.Draw("SAME")
                            res['1d_lambdas'].append(lam_peak)
                            res['1d_funcs'].append((None, f1d_peak))
                
            if show_fit_params:
                # Add stats box for shared params like P-value, sigma, tau
                stats_box = ROOT.TPaveText(0.65, 0.45, 0.88, 0.88, "NDC")
                stats_box.SetFillColor(ROOT.kWhite)
                stats_box.SetBorderSize(1)
                stats_box.SetTextAlign(12) 
                
                prob = fit_res.Prob()
                chi2_ndf = fit_res.Chi2() / fit_res.Ndf() if fit_res.Ndf() > 0 else 0
                stats_box.AddText(f"P-value: {prob:.4g}")
                stats_box.AddText(f"#chi^{{2}}/ndf: {chi2_ndf:.2f}")
                
                for i in range(fit_res.NPar()):
                    p_name = f_to_fit_2d.GetParName(i)
                    # We might only want to show shared parameters (like mu, sigma, tau) rather than all 100 amplitudes
                    if not p_name.startswith("amplitude") and not p_name.startswith("bg_const") and not p_name.startswith("bg_slope") and not p_name.startswith("bg_shift"):
                        p_val = fit_res.Parameter(i)
                        p_err = fit_res.ParError(i)
                        stats_box.AddText(f"{p_name}: {p_val:.4g} #pm {p_err:.4g}")
                        
                stats_box.Draw("SAME")
                res['multi_stats_box'] = stats_box

            pad2.cd()
            pad2.SetGridy()
            
            # Draw residuals for all spectra
            for j, spec in enumerate(self.spectra):
                color = colors[j % len(colors)]
                
                resid_graph = ROOT.TGraphErrors()
                resid_graph.SetMarkerColor(color)
                resid_graph.SetLineColor(color)
                resid_graph.SetMarkerStyle(20)
                resid_graph.SetMarkerSize(0.6)
                
                pt_idx = 0
                bin_start = spec.FindBin(e_low)
                bin_end = spec.FindBin(e_high)
                for bin_i in range(bin_start, bin_end + 1):
                    x_val = spec.GetBinCenter(bin_i)
                    y_val = spec.GetBinContent(bin_i)
                    y_err = spec.GetBinError(bin_i)
                    
                    fit_y = f_to_fit_2d.Eval(x_val, j)
                    resid = y_val - fit_y
                    
                    resid_graph.SetPoint(pt_idx, x_val, resid)
                    resid_graph.SetPointError(pt_idx, 0, y_err)
                    pt_idx += 1
                    
                if j == 0:
                    resid_graph.Draw("AP")
                    resid_graph.GetXaxis().SetLabelSize(0.1)
                    resid_graph.GetXaxis().SetTitleSize(0.12)
                    resid_graph.GetYaxis().SetLabelSize(0.1)
                    resid_graph.GetYaxis().SetTitleSize(0.1)
                    resid_graph.GetYaxis().SetTitleOffset(0.4)
                    resid_graph.GetYaxis().SetTitle("Data - Fit")
                    resid_graph.GetYaxis().SetNdivisions(505)
                else:
                    resid_graph.Draw("P SAME")
                    
                if 'resid_graphs' not in res:
                    res['resid_graphs'] = []
                res['resid_graphs'].append(resid_graph)
                
            new_canvas.Draw()
            new_canvas.Update()
            ROOT.SetOwnership(new_canvas, False)
            res['display_canvas_multi'] = new_canvas
            res['pad1'] = pad1
            res['pad2'] = pad2
        else:
            print(f"Invalid peak index: {peak_index}")

    def save(self, filepath):
        '''
        Save fit peak locations and parameters with uncertainties to {filepath}.csv
        (ROOT file saving for 2D simultaneous fits is not yet fully supported).
        '''
        if not filepath.endswith('.root'):
            filepath += '.root'
        csv_filepath = filepath[:-5] + '.csv'
        print(f"Saving multi_spectrum_fitter CSV to {csv_filepath}...")
        
        with open(csv_filepath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Identify all unique free parameter base names
            free_base_params = set()
            for res in self.fit_results:
                if res is None or 'f_to_fit_2d' not in res or 'fit_res' not in res: continue
                fit_res = res['fit_res']
                f_to_fit = res['f_to_fit_2d']
                
                for j in range(f_to_fit.GetNpar()):
                    param_free = True
                    try:
                        if hasattr(fit_res, "Get") and fit_res.Get():
                            param_free = not fit_res.Get().IsParameterFixed(j)
                        elif hasattr(fit_res, "IsParameterFixed"):
                            param_free = not fit_res.IsParameterFixed(j)
                        else:
                            param_free = (fit_res.ParError(j) != 0.0)
                    except Exception:
                        param_free = (fit_res.ParError(j) != 0.0)
                    
                    if param_free:
                        name = f_to_fit.GetParName(j)
                        # amplitude_{peak}_{spec}
                        m3 = re.match(r'^(.*)_(\d+)_(\d+)$', name)
                        if m3:
                            free_base_params.add(m3.group(1) + "_" + m3.group(3))
                            continue
                        
                        m2 = re.match(r'^(.*)_(\d+)$', name)
                        if m2:
                            # could be mu_{peak} or bg_const_{spec}
                            # we treat bg_const as global, mu as peak-specific
                            base = m2.group(1)
                            if base in ['bg_const', 'bg_slope', 'bg_shift']:
                                free_base_params.add(name)
                            else:
                                free_base_params.add(base)
                        else:
                            free_base_params.add(name)
            
            sorted_params = sorted(list(free_base_params))
            priorities = {'mu': 1, 'sigma': 2, 'tau': 3}
            sorted_params.sort(key=lambda x: (priorities.get(x.split('_')[0], 100), x))
            
            header = ['fit_index', 'loc_guess', 'p_value']
            for p in sorted_params:
                header.extend([f'{p}_val', f'{p}_err'])
            csvwriter.writerow(header)
            
            for i, res in enumerate(self.fit_results):
                if res is None or 'f_to_fit_2d' not in res or 'fit_res' not in res: continue
                
                loc_guesses = self.peaks_to_fit[i][0]
                if not isinstance(loc_guesses, (list, tuple, np.ndarray)):
                    loc_guesses = [loc_guesses]
                
                fit_res = res['fit_res']
                f_to_fit = res['f_to_fit_2d']
                p_value = fit_res.Prob()
                
                # pre-extract all param values and errors
                p_vals = {}
                p_errs = {}
                for j in range(f_to_fit.GetNpar()):
                    name = f_to_fit.GetParName(j)
                    p_vals[name] = fit_res.Parameter(j)
                    p_errs[name] = fit_res.ParError(j)
                
                for k, loc in enumerate(loc_guesses):
                    row_dict = {}
                    
                    for name, val in p_vals.items():
                        err = p_errs[name]
                        
                        m3 = re.match(r'^(.*)_(\d+)_(\d+)$', name)
                        if m3:
                            base = m3.group(1)
                            pk_idx = int(m3.group(2))
                            spec_idx = m3.group(3)
                            if pk_idx == k:
                                row_dict[f"{base}_{spec_idx}_val"] = val
                                row_dict[f"{base}_{spec_idx}_err"] = err
                            continue
                            
                        m2 = re.match(r'^(.*)_(\d+)$', name)
                        if m2:
                            base = m2.group(1)
                            idx = int(m2.group(2))
                            if base in ['bg_const', 'bg_slope', 'bg_shift']:
                                row_dict[name + "_val"] = val
                                row_dict[name + "_err"] = err
                            else:
                                if idx == k:
                                    row_dict[base + "_val"] = val
                                    row_dict[base + "_err"] = err
                            continue
                            
                        # global par (like shared sigma, tau)
                        row_dict[name + "_val"] = val
                        row_dict[name + "_err"] = err
                        
                    row = [i, loc, p_value]
                    for p in sorted_params:
                        row.append(row_dict.get(p + '_val', ''))
                        row.append(row_dict.get(p + '_err', ''))
                    csvwriter.writerow(row)
        
        # Save to ROOT file
        f = ROOT.TFile(filepath, "RECREATE")
        
        for idx, spec in enumerate(self.spectra):
            spec.Write(f"spectrum_{idx}")
            
        python_state = {
            'is_multi': True,
            'num_spectra': len(self.spectra),
            'peak_model': self.peak_model,
            'peaks_to_fit': self.peaks_to_fit,
            'fit_options': self.fit_options,
            'num_fit_results': len(self.fit_results),
            'parameterizations': self.parameterizations,
            'shared_sigma': getattr(self, 'shared_sigma', False),
            'location_wiggle': getattr(self, 'location_wiggle', 10)
        }
        
        import inspect
        import dill
        source_dict = {}
        for k, v in self.param_bound_functions.items():
            try:
                source_dict[k] = inspect.getsource(v).strip()
            except Exception:
                source_dict[k] = str(v)
        python_state['param_bound_functions'] = source_dict
        
        for i, res in enumerate(self.fit_results):
            if res is None: continue
            
            if res.get('fit_res') and res['fit_res'].Get():
                res['fit_res'].Get().Write(f"peak_{i}_fit_res")
                
            if res.get('sub_hist_2d'): res['sub_hist_2d'].Write(f"peak_{i}_sub_hist_2d")
            if res.get('f_to_fit_2d'): res['f_to_fit_2d'].Write(f"peak_{i}_f_to_fit_2d")
            if res.get('h_fit_2d'): res['h_fit_2d'].Write(f"peak_{i}_h_fit_2d")
            if res.get('h_resid_2d'): res['h_resid_2d'].Write(f"peak_{i}_h_resid_2d")
            if res.get('canvas_2d'): res['canvas_2d'].Write(f"peak_{i}_canvas_2d")
            if res.get('pm'):
                python_state[f'peak_{i}_pm_names'] = res['pm'].names
                if hasattr(res['pm'], 'bg_func_name'):
                    python_state[f'peak_{i}_bg_func_name'] = res['pm'].bg_func_name
                if hasattr(res['pm'], 'peak_func_name'):
                    python_state[f'peak_{i}_peak_func_name'] = res['pm'].peak_func_name
                if hasattr(res['pm'], 'cpp_code'):
                    python_state[f'peak_{i}_cpp_code'] = res['pm'].cpp_code
            
        pickled_hex_string = dill.dumps(python_state).hex()
        obj_string = ROOT.TObjString(pickled_hex_string)
        obj_string.Write("python_state")
        
        f.Close()
        print("ROOT save complete.")