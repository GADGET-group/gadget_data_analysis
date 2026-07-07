def show_init_cal_comparison(fit_name):
    '''
    Create and show a TGraph comparing the true energy to fit mu for each peak in peak_fitting/gamma_peaks.csv
    where "use in cal" is true. Fit values are pulled from peak_fitting/{fit_name}.csv. Plot a line with slope
    of one and intercept of 0. In smaller axis below the main scatter plot, show (true energy) - (fit energy).
    '''