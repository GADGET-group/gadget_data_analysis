import matplotlib.pyplot as plt
import final_gamma_ecal

def build_level_scheme(transitions, gamma_fit_name):
    '''
    transitions: List of dicts [{'Ei': float, 'Ef': float, 'I': float (optional), 'E': float (specifies location guess)}]
    The Ei and Ef are just labels and not necessarily the correct energy which will be determined when the decay scheme is buit,
    with the exception of the special 0 level which is the ground state.
    Starting from the ground state, calculate the energy of all connected levels to build up the decay scheme. The nenergy of each level
    can be calculated using final_gamma_ecal.apply_energy_calibration_to_cascade. 
    If more than one cascade connects a level to the ground state, use the one which has the smallest uncertainty. 
    Apply_energy_calibration to cascade can also be used to get the energy for an individual gamma ray.
    final_gamma_ecal.get_fit_results can be used to convert from peak location guess to the inputs needed for apply_energy_calibraiton
    '''
    adj = {}
    levels = set()
    for trans in transitions:
        u = trans['Ei']
        v = trans['Ef']
        e = trans['E']
        if u not in adj:
            adj[u] = []
        adj[u].append((v, e))
        levels.add(u)
        levels.add(v)

    def get_all_paths(u, target, current_path, all_paths, visited):
        if u == target:
            all_paths.append(list(current_path))
            return
        visited.add(u)
        if u in adj:
            for v, e in adj[u]:
                if v not in visited:
                    current_path.append(e)
                    get_all_paths(v, target, current_path, all_paths, visited)
                    current_path.pop()
        visited.remove(u)

    true_level_E = {}
    true_level_err = {}

    for L in levels:
        if L == 0 or L == 0.0:
            true_level_E[L] = 0.0
            true_level_err[L] = 0.0
            continue
            
        all_paths = []
        get_all_paths(L, 0.0, [], all_paths, set())
        
        best_e = L
        best_err = float('inf')
        
        for path in all_paths:
            fit_vals, fit_errs = final_gamma_ecal.get_fit_results(path, gamma_fit_name)
            if fit_vals is not None:
                e_cal, e_err = final_gamma_ecal.apply_energy_calibration_to_cascade(fit_vals, fit_errs, final_gamma_ecal.calibration)
                if e_err < best_err:
                    best_err = e_err
                    best_e = e_cal
                    
        if best_err != float('inf'):
            true_level_E[L] = best_e
            true_level_err[L] = best_err
        else:
            # Fallback if no path to 0 or fits failed
            true_level_E[L] = float(L)
            true_level_err[L] = 0.0

    levels_out = {}
    for L, e_cal in true_level_E.items():
        levels_out[e_cal] = ('', true_level_err[L])

    transitions_out = []
    for trans in transitions:
        new_trans = trans.copy()
        new_trans['Ei'] = true_level_E[trans['Ei']]
        new_trans['Ef'] = true_level_E[trans['Ef']]
        
        fit_vals, fit_errs = final_gamma_ecal.get_fit_results([trans['E']], gamma_fit_name)
        if fit_vals is not None:
            e_cal, e_err = final_gamma_ecal.apply_energy_calibration_to_cascade(fit_vals, fit_errs, final_gamma_ecal.calibration)
            new_trans['E'] = e_cal
            new_trans['dE'] = e_err
        else:
            new_trans['E'] = trans['E']
            new_trans['dE'] = 0.0
            
        transitions_out.append(new_trans)

    return levels_out, transitions_out

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

    # Define gamma transitions
    my_transitions = [
        {'Ei': 1004, 'Ef': 0.0, 'E': 1003.22},
        {'Ei': 2192, 'Ef': 1004, 'E': 1188},
        {'Ei': 3394, 'Ef': 2192, 'E': 1202},
        {'Ei': 3394, 'Ef': 1004, 'E': 2390},
        {'Ei': 3394, 'Ef': 0.0, 'E': 3394},
        {'Ei': 5722, 'Ef': 1004, 'E': 4718},
        {'Ei': 5722, 'Ef': 0.0, 'E': 5722},
        {'Ei': 4039, 'Ef': 3010, 'E': 1028},
        {'Ei': 3010, 'Ef': 1004, 'E': 2006},
        {'Ei': 4039, 'Ef': 2558, 'E': 1481},
        {'Ei': 2558, 'Ef': 1004, 'E': 1554},
        {'Ei': 2558, 'Ef': 0.0, 'E': 2558},
        {'Ei': 3999, 'Ef': 0.0, 'E': 3999},
        {'Ei': 3999, 'Ef': 2558, 'E': 1441},
        {'Ei': 3999, 'Ef': 1004, 'E': 2995},
        {'Ei': 4850, 'Ef': 2558, 'E': 2292},
        {'Ei': 4850, 'Ef': 1004, 'E': 3846},
        {'Ei': 4850, 'Ef': 3437, 'E': 1413},
        {'Ei': 3437, 'Ef': 1004, 'E': 2433},
        {'Ei': 4850, 'Ef': 0.0, 'E': 4850},
        {'Ei': 5807, 'Ef': 1004, 'E': 4803},
        {'Ei': 5807, 'Ef': 0.0, 'E': 5807},
        {'Ei': 5048, 'Ef': 0.0, 'E': 5048},
        {'Ei': 5296, 'Ef': 0.0, 'E': 5296},
        {'Ei': 5266, 'Ef': 0.0, 'E': 5266},
        {'Ei': 5447, 'Ef': 0.0, 'E': 5447},
        {'Ei': 5559, 'Ef': 0.0, 'E': 5559}
    ]

    my_levels, my_transitions_cal = build_level_scheme(my_transitions, '60Ga_all_gamma')
    draw_level_scheme(my_levels, my_transitions_cal, title="$^{60}$Zn Level Scheme")