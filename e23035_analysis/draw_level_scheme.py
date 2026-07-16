import matplotlib.pyplot as plt

def draw_level_scheme(levels, transitions, title="Nuclear Level Scheme"):
    """
    Draws a nuclear level scheme using Matplotlib with anti-overlap label spacing.
    
    Parameters:
    levels (dict): Dictionary of {energy: 'Jpi'} or {energy: ('Jpi', error)}
    transitions (list): List of dicts [{'Ei': float, 'Ef': float, 'I': float (optional), 'E': float (optional), 'dE': float (optional)}]
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # 1. Setup Spacing and Dimensions
    n_transitions = len(transitions)
    x_start = 0
    x_end = max(4, n_transitions + 2)  # Width depends on number of arrows
    
    # Extract max intensity for normalizing arrow widths
    valid_intensities = [t['I'] for t in transitions if 'I' in t and t['I'] is not None]
    max_intensity = max(valid_intensities) if valid_intensities else 1.0

    # 2. Anti-Overlap Label Algorithm
    # Iteratively push labels apart if they are closer than min_spacing
    energies = sorted(levels.keys())
    label_y = {e: e for e in energies}
    max_e = max(energies) if energies else 1
    
    # Define minimum vertical distance between labels (e.g., 2.5% of total plot height)
    min_spacing = max_e * 0.025 

    for _ in range(20):  # 20 iterations to let the positions "relax"
        for i in range(len(energies) - 1):
            e1, e2 = energies[i], energies[i+1]
            y1, y2 = label_y[e1], label_y[e2]
            
            if y2 - y1 < min_spacing:
                shift = (min_spacing - (y2 - y1)) / 2.0
                label_y[e1] -= shift
                label_y[e2] += shift

    # 3. Draw Energy Levels and Labels
    for energy, val in levels.items():
        if isinstance(val, (tuple, list)):
            jpi = val[0]
            err = val[1]
        elif isinstance(val, dict):
            jpi = val.get('jpi', '')
            err = val.get('err', None)
        else:
            jpi = val
            err = None
            
        y_pos = label_y[energy] # The newly calculated, non-overlapping Y position
        
        # Draw the horizontal energy line at its TRUE mathematical position
        ax.hlines(y=energy, xmin=x_start, xmax=x_end, color='black', linewidth=2)
        
        # Is the label shifted significantly? (Tolerance of 0.1% to draw leader lines)
        is_shifted = abs(y_pos - energy) > (max_e * 0.001)

        # Label the Spin-Parity (Jpi) ONLY if it contains text
        if jpi:
            ax.text(x_start - 0.2, y_pos, f"${jpi}$", 
                    va='center', ha='right', fontsize=14, color='darkred')
            # Draw dotted leader line from text to true level
            if is_shifted:
                ax.plot([x_start - 0.2, x_start], [y_pos, energy], color='gray', lw=1, ls=':')
        
        # Label the Energy on the right
        if err is not None:
            err_nndc = int(round(err * 100))
            energy_text = f"{energy:.2f}({err_nndc})"
        else:
            energy_text = f"{energy:.2f}"
            
        ax.text(x_end + 0.2, y_pos, energy_text, 
                va='center', ha='left', fontsize=12)
        # Draw dotted leader line from level to text
        if is_shifted:
            ax.plot([x_end, x_end + 0.2], [energy, y_pos], color='gray', lw=1, ls=':')

    # 4. Draw Transitions (Gamma Rays)
    transitions_sorted = sorted(transitions, key=lambda x: x['Ei'], reverse=True)
    
    for i, trans in enumerate(transitions_sorted):
        ei = trans['Ei']
        ef = trans['Ef']
        intensity = trans.get('I', None)
        
        e_gamma = trans.get('E', ei - ef)
        e_err = trans.get('dE', None)
        
        x_pos = x_start + 1 + i 
        
        # Calculate arrow thickness
        if intensity is not None and max_intensity > 0:
            arrow_width = max(1.0, (intensity / max_intensity) * 6.0)
        else:
            arrow_width = 2.0
            
        ax.annotate('', 
                    xy=(x_pos, ef), xycoords='data',      
                    xytext=(x_pos, ei), textcoords='data', 
                    arrowprops=dict(arrowstyle="-|>", color='blue', lw=arrow_width, mutation_scale=15))
        
        if e_err is not None:
            err_nndc = int(round(e_err * 100))
            gamma_text = f"{e_gamma:.2f}({err_nndc})"
        else:
            gamma_text = f"{e_gamma:.2f}"
            
        ax.text(x_pos - 0.15, (ei + ef) / 2.0, gamma_text, 
                rotation=90, va='center', ha='right', fontsize=10)
        
        if intensity is not None:
            ax.text(x_pos, ei + (max_e * 0.02), f"{intensity}", 
                    va='bottom', ha='center', fontsize=9, color='gray')

    # 5. Final Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off') 
    
    # Auto-scale the y-axis using the pushed label coordinates so text isn't cut off
    max_label_y = max(label_y.values())
    ax.set_ylim(-max_e * 0.05, max_label_y * 1.05)
    ax.set_xlim(x_start - 1.5, x_end + 1.5)
    
    plt.tight_layout()
    #plt.savefig(f"{title.replace(' ', '_').replace('$', '').replace('^', '')}.pdf", format='pdf', bbox_inches='tight')
    #plt.close()
    plt.show()

# ==========================================
# Example Usage 
# ==========================================
if __name__ == "__main__":
    # Define energy levels in keV and their Jpi assignments
    my_levels = {
        0.0: "0^+",
        1003.22: ('2^+', 0.16),
        2191.59: ('', 0.16),
        2557.85: ('', 0.17),
        3010.31: ('', 0.20),
        3393.74: ('', 0.20),
        3436.56: ('', 0.17),
        3999.26: ('', 0.24),
        4038.70: ('', 0.27),
        4850.41: ('', 0.31),
        4891.18: ('', 0.33),
        5183.00: ('', 0.50),
        5721.81: ('', 0.74),
        5806.86: ('', 0.39),
        5048.18: ('', 0.39),
        5265.85: ('', 0.53),
        5296.60: ('', 0.39),
        5446.88: ('', 0.89),
        5558.75: ('', 0.39)
    }

    # Define gamma transitions
    my_transitions = [
        {'Ei': 1003.22, 'Ef': 0.0, 'E': 1003.22, 'dE': 0.16},
        {'Ei': 2191.59, 'Ef': 1003.22, 'E': 1188.37, 'dE': 0.16},
        {'Ei': 3393.74, 'Ef': 2191.59, 'E': 1202.10, 'dE': 0.23},
        {'Ei': 3393.74, 'Ef': 1003.22, 'E': 2390.27, 'dE': 0.18},
        {'Ei': 3393.74, 'Ef': 0.0, 'E': 3393.74, 'dE': 0.20},
        {'Ei': 5721.81, 'Ef': 1003.22, 'E': 4717.87, 'dE': 0.30},
        {'Ei': 5721.81, 'Ef': 0.0, 'E': 5721.81, 'dE': 0.74},
        {'Ei': 4038.70, 'Ef': 3010.31, 'E': 1028.10, 'dE': 0.27},
        {'Ei': 3010.31, 'Ef': 1003.22, 'E': 2007.09, 'dE': 0.20},
        {'Ei': 4038.70, 'Ef': 2557.85, 'E': 1481.14, 'dE': 0.16},
        {'Ei': 2557.85, 'Ef': 1003.22, 'E': 1554.47, 'dE': 0.15},
        {'Ei': 2557.85, 'Ef': 0.0, 'E': 2557.85, 'dE': 0.17},
        {'Ei': 3999.26, 'Ef': 0.0, 'E': 3999.26, 'dE': 0.24},
        {'Ei': 3999.26, 'Ef': 2557.85, 'E': 1441.70, 'dE': 0.15},
        {'Ei': 3999.26, 'Ef': 1003.22, 'E': 2996.34, 'dE': 0.19},
        {'Ei': 4850.41, 'Ef': 2557.85, 'E': 2292.53, 'dE': 0.16},
        {'Ei': 4850.41, 'Ef': 1003.22, 'E': 3846.89, 'dE': 0.23},
        {'Ei': 4850.41, 'Ef': 3436.56, 'E': 1413.27, 'dE': 0.18},
        {'Ei': 3436.56, 'Ef': 1003.22, 'E': 2433.34, 'dE': 0.17},
        {'Ei': 4850.41, 'Ef': 0.0, 'E': 4850.41, 'dE': 0.31},
        {'Ei': 5806.86, 'Ef': 1003.22, 'E': 4803.73, 'dE': 0.30},
        {'Ei': 5806.86, 'Ef': 0.0, 'E': 5806.86, 'dE': 0.39},
        {'Ei': 5048.18, 'Ef': 0.0, 'E': 5048.18, 'dE': 0.39},
        {'Ei': 5296.60, 'Ef': 0.0, 'E': 5296.60, 'dE': 0.39},
        {'Ei': 5265.85, 'Ef': 0.0, 'E': 5265.85, 'dE': 0.53},
        {'Ei': 5446.88, 'Ef': 0.0, 'E': 5446.88, 'dE': 0.89},
        {'Ei': 5558.75, 'Ef': 0.0, 'E': 5558.75, 'dE': 0.39}
    ]

    draw_level_scheme(my_levels, my_transitions, title="$^{60}$Zn Level Scheme")