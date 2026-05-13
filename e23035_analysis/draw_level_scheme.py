import matplotlib.pyplot as plt

def draw_level_scheme(levels, transitions, title="Nuclear Level Scheme"):
    """
    Draws a nuclear level scheme using Matplotlib with anti-overlap label spacing.
    
    Parameters:
    levels (dict): Dictionary of {energy: 'Jpi'}
    transitions (list): List of dicts [{'Ei': float, 'Ef': float, 'I': float (optional)}]
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
    for energy, jpi in levels.items():
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
        ax.text(x_end + 0.2, y_pos, f"{energy:.0f}", 
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
        e_gamma = ei - ef
        
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
        
        ax.text(x_pos - 0.15, (ei + ef) / 2.0, f"{e_gamma:.0f}", 
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
        1004:'2^+',
        2192: '',
        2557: '',
        3011: '',
        3393: '',
        3437: '',
        4000: '',
        4038: '',
        4852: '',
        4892: '',
        5183: '',
        5723: '',
        5809: ''
    }

    # Define gamma transitions
    my_transitions = [
        {'Ei': 1004, 'Ef': 0.0},#'I': 100
        {'Ei': 2192, 'Ef': 1004},
        {'Ei':3393, 'Ef': 2192},
        {'Ei':3393, 'Ef': 1004},
        {'Ei':3393, 'Ef': 0},
        {'Ei':5723, 'Ef':1004},
        {'Ei':5723, 'Ef':0 },
        {'Ei':4038, 'Ef':3011 },
        {'Ei':3011, 'Ef':1004 },
        {'Ei':4038, 'Ef':2557 },
        {'Ei':2557, 'Ef':1004 },
        {'Ei':2557, 'Ef':0 },
        {'Ei':4000, 'Ef':0 },
        {'Ei':4000, 'Ef': 2557},
        {'Ei':4000, 'Ef': 1004},
        {'Ei':4852, 'Ef':2557 },
        {'Ei':4852, 'Ef':1004 },
        {'Ei':4852, 'Ef':3437 },
        {'Ei':3437, 'Ef': 1004},
        {'Ei':4852, 'Ef':0 },
        {'Ei':5809, 'Ef':1004 },
        {'Ei':5809, 'Ef': 0},
    ]

    draw_level_scheme(my_levels, my_transitions, title="$^{60}$Zn Level Scheme")