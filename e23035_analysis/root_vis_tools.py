import uuid

import ROOT

from raw_viewer import ddas_interface

experiment = 'e23035'

def draw_overlaid_histograms(hist_dict, title="Overlaid Histograms", x_label="X", y_label="Counts"):
    if not hist_dict:
        return None, None, None

    unique_id = uuid.uuid4().hex[:8]
    canvas = ROOT.TCanvas(f"c_stack_{unique_id}", title, 800, 600)
    canvas.SetGridx()
    canvas.SetGridy()

    # 1. Create the THStack
    stack = ROOT.THStack(f"stack_{unique_id}", title)
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)

    colors = [ROOT.kBlack, ROOT.kRed+1, ROOT.kBlue+1, ROOT.kGreen+2, 
              ROOT.kMagenta+1, ROOT.kOrange+7, ROOT.kCyan+2, ROOT.kViolet+2]

    # 2. Add histograms to the stack
    for i, (label, hist) in enumerate(hist_dict.items()):
        color = colors[i % len(colors)]
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        hist.SetStats(0) 
        
        stack.Add(hist)
        legend.AddEntry(hist, label, "l")

    # 3. Draw with "nostack"
    # "nostack" ensures they overlay rather than physically stacking on top of each other.
    # "hist" forces them to draw as lines rather than points with errors.
    stack.Draw("nostack hist")

    # 4. Set Axis Titles (CRITICAL: Must be done AFTER Draw)
    # ROOT doesn't build the underlying histogram for the stack axes until Draw() is called.
    stack.GetXaxis().SetTitle(x_label)
    stack.GetYaxis().SetTitle(y_label)

    legend.Draw()
    canvas.Update()

    # Must return the stack as well to keep it alive in memory!
    return canvas, legend, stack

def create_2d_hist_from_dict(hist_dict, title="Stacked 2D Histogram", y_label=""):
    """
    Takes a single dictionary of {label: TH1D} and stacks them along the Y-axis 
    to create a single 2D histogram.
    """
    if not hist_dict:
        print("Error: The dictionary is empty.")
        return None, None

    # 1. Extract X-axis properties from the first histogram
    # We assume all 1D histograms in the dictionary share the same X-axis binning.
    first_hist = list(hist_dict.values())[0]
    n_bins_x = first_hist.GetNbinsX()
    x_min = first_hist.GetXaxis().GetXmin()
    x_max = first_hist.GetXaxis().GetXmax()
    
    # Try to grab the X-axis title from the original histogram
    x_label = first_hist.GetXaxis().GetTitle()
    if not x_label:
        x_label = "X"

    n_bins_y = len(hist_dict)

    # 2. Create the TH2D and Canvas
    unique_id = uuid.uuid4().hex[:8]
    h2_name = f"h2_stack_{unique_id}"
    h2_title = f"{title};{x_label};{y_label};Counts"
    
    th2 = ROOT.TH2D(h2_name, h2_title, n_bins_x, x_min, x_max, n_bins_y, 0, n_bins_y)
    
    canvas = ROOT.TCanvas(f"c_2d_{unique_id}", title, 800, 600)
    
    # Add margins so long Y-axis labels and the Z-axis color bar don't get cut off
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)

    # 3. Fill the 2D histogram
    for y_idx, (key, h1) in enumerate(hist_dict.items()):
        # ROOT bins are 1-indexed
        y_bin = y_idx + 1 
        
        # Set the Y-axis bin label to the dictionary key for easy reading
        th2.GetYaxis().SetBinLabel(y_bin, str(key))
        
        # Loop over the X bins and copy content
        for x_bin in range(1, n_bins_x + 1):
            content = h1.GetBinContent(x_bin)
            error = h1.GetBinError(x_bin)
            
            th2.SetBinContent(x_bin, y_bin, content)
            th2.SetBinError(x_bin, y_bin, error)

    # 4. Draw the result
    th2.SetStats(0)          # Turn off the stats box
    th2.Draw("COLZ")         # Draw with the Z-axis color palette
    canvas.Update()

    return canvas, th2

def plot_crystal_vs_time(ddas_run, clover, seconds_per_tbin=360):
    """
    Retrieves and plots a 2D histogram of Energy vs Time for a specific clover crystal.
    
    Args:
        ddas_run: The run object/data to analyze.
        clover (str): The crystal identifier (e.g., '3c', '1a').
        
    Returns:
        canvas, histogram (to prevent garbage collection)
    """
    # 1. Get run times using the module
    t_start, t_stop = ddas_interface.get_first_and_last_ddas_time(experiment, ddas_run)
    
    # Calculate time bins (1 bin per 360 seconds = 6 minutes)
    t_bins = int((t_stop - t_start) / seconds_per_tbin)
    if t_bins <= 0:
        t_bins = 1 
        
    # 2. Construct dynamic naming strings based on the clover argument
    unique_id = uuid.uuid4().hex[:6]
    hist_name = f'run{ddas_run}_crystal_{clover}_vs_time_{unique_id}'
    hist_title = f'Run {ddas_run} Clover {clover} Energy vs Time'
    
    # Dynamically format the draw string for this specific crystal
    draw_string = f'clover_{clover}_t:clover_{clover}_e'
    
    # 3. Fetch the histogram
    # Binning format: (x_bins, x_min, x_max, y_bins, y_min, y_max)
    binning = (300, 0, 3000, t_bins, t_start, t_stop)
    
    hist_vs_t = ddas_interface.get_histogram(
        experiment,
        ddas_run, 
        binning,
        hist_name, 
        hist_title, 
        draw_string
    )
    
    # 4. Create canvas and draw
    canvas_name = f"c_{clover}_vs_t_{unique_id}"
    canvas = ROOT.TCanvas(canvas_name, hist_title, 800, 600)
    
    hist_vs_t.SetStats(0)
    hist_vs_t.GetXaxis().SetTitle("Energy")
    hist_vs_t.GetYaxis().SetTitle("Time (s)")
    
    hist_vs_t.Draw('colz')
    
    # Apply LogZ to the canvas
    canvas.SetLogz(1)
    canvas.Update()
    
    return canvas, hist_vs_t

def label_peaks(hist, peaks, y_offset_factor=1.05, text_size=0.03, color=ROOT.kBlack, angle=90):
    """
    Labels peaks on a 1D ROOT histogram.

    Args:
        hist: The ROOT TH1 object to label.
        peaks: A list of tuples, where each tuple is (label_string, x_location).
        y_offset_factor: Factor by which to multiply the bin content to set the Y position of the label.
        text_size: Size of the text.
        color: Color of the text.
        angle: Angle of the text in degrees.

    Returns:
        A list of ROOT.TLatex objects. This list MUST be kept alive in memory 
        for the labels to remain visible on the canvas.
    """
    latex_labels = []
    for label, x_loc in peaks:
        bin_num = hist.GetXaxis().FindBin(x_loc)
        y_val = hist.GetBinContent(bin_num)
        
        latex = ROOT.TLatex(x_loc, y_val * y_offset_factor, str(label))
        latex.SetTextSize(text_size)
        latex.SetTextColor(color)
        latex.SetTextAngle(angle)
        latex.SetTextAlign(12) 
        
        latex.Draw()
        latex_labels.append(latex)
        
    return latex_labels

    