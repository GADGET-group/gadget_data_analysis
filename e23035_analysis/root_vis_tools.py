import ROOT
import uuid

def draw_overlaid_histograms(hist_dict, title="Overlaid Histograms", x_label="keV", y_label="Counts"):
    """
    Takes a dictionary of {label: TH1} and draws them on a single canvas 
    with different colors and a legend.
    """
    if not hist_dict:
        print("Error: Empty dictionary provided.")
        return None, None

    # 1. Generate unique canvas to prevent overwriting
    unique_id = uuid.uuid4().hex[:8]
    canvas = ROOT.TCanvas(f"c_overlaid_{unique_id}", title, 800, 600)
    
    # Optional: add a grid for readability
    canvas.SetGridx()
    canvas.SetGridy()

    # 2. Setup the Legend (X1, Y1, X2, Y2 in normalized coordinates 0.0 to 1.0)
    # Placing it in the top right corner
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0) # Cleaner look without a heavy border
    legend.SetFillStyle(0)  # Transparent background

    # 3. Define a palette of distinct, colorblind-friendly-ish ROOT colors
    colors = [
        ROOT.kBlack, 
        ROOT.kRed + 1, 
        ROOT.kBlue + 1, 
        ROOT.kGreen + 2, 
        ROOT.kMagenta + 1, 
        ROOT.kOrange + 7, 
        ROOT.kCyan + 2, 
        ROOT.kViolet + 2
    ]

    # 4. Find the global maximum to scale the Y-axis correctly
    global_max = 0
    for hist in hist_dict.values():
        current_max = hist.GetMaximum()
        if current_max > global_max:
            global_max = current_max

    # 5. Drawing Loop
    is_first = True
    for i, (label, hist) in enumerate(hist_dict.items()):
        # Assign color (modulo allows looping if you have more hists than colors)
        color = colors[i % len(colors)]
        
        hist.SetLineColor(color)
        hist.SetLineWidth(2)
        
        # Turn off the stats box (multiple stats boxes overlapping looks messy)
        hist.SetStats(0) 
        
        # Add to legend ("l" means draw a line in the legend)
        legend.AddEntry(hist, label, "l")

        if is_first:
            # Scale the first histogram so everything fits (add 15% headroom)
            hist.SetMaximum(global_max * 1.15)
            
            # Set titles
            hist.SetTitle(title)
            hist.GetXaxis().SetTitle(x_label)
            hist.GetYaxis().SetTitle(y_label)
            
            # Draw first hist normally
            hist.Draw("HIST") 
            is_first = False
        else:
            # Draw subsequent hists with "SAME" so they overlay
            hist.Draw("HIST SAME")

    # 6. Draw legend and update canvas
    legend.Draw()
    canvas.Update()

    # Return canvas and legend to prevent Python from garbage collecting them!
    return canvas, legend