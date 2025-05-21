import os
import subprocess
import configparser
import csv
import shutil
import sys
sys.path.append("/mnt/projects/e21072/OfflineAnalysis/analysis_scripts/joe/gadget_analysis")

import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.messagebox

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
from skspatial.objects import Line, Point

from tqdm import tqdm

from fit_gui.HistogramFitFrame import HistogramFitFrame
from raw_viewer import raw_h5_file
from raw_viewer import heritage_h5_file

def main():
    # make sure that the paths passed to this script are the full paths including the base names (the filename)
    file_path = sys.argv[1]
    flat_lookup_path = sys.argv[2]
    settings_path = sys.argv[3]

    data = raw_h5_file.raw_h5_file(file_path, flat_lookup_csv=flat_lookup_path, zscale=0.65)

    config = configparser.ConfigParser()
    config.read(settings_path)
    data.length_ic_threshold = config.get('ttk.Entry','length_ic_threshold')
    data.energy_ic_threshold = config.get('ttk.Entry','energy_ic_threshold')
    data.view_threshold = config.get('ttk.Entry','view_threshold')
    data.include_cobos = config.get('ttk.Entry','include_cobos')
    data.include_asads = config.get('ttk.Entry','include_asads')
    data.include_pads = config.get('ttk.Entry','include_pads')
    data.veto_threshold = config.get('ttk.Entry','veto_threshold')
    data.range_min = config.get('ttk.Entry','range_min')
    data.range_max = config.get('ttk.Entry','range_max')
    data.min_ic = config.get('ttk.Entry','min_ic')
    data.max_ic = config.get('ttk.Entry','max_ic')
    data.angle_min = config.get('ttk.Entry','angle_min')
    data.angle_max = config.get('ttk.Entry','angle_max')
    data.background_bin_start = config.get('ttk.Entry','background_bin_start')
    data.background_bin_stop = config.get('ttk.Entry','background_bin_stop')
    data.zscale = config.get('ttk.Entry','zscale')
    data.near_peak_window_width = config.get('ttk.Entry','near_peak_window_width')
    data.peak_first_allowed_bin = config.get('ttk.Entry','peak_first_allowed_bin')
    data.peak_last_allowed_bin = config.get('ttk.Entry','peak_last_allowed_bin')
    data.peak_mode = config.get('ttk.OptionMenu','peak_mode')
    data.background_mode = config.get('ttk.OptionMenu','background_mode')
    data.remove_outliers = config.get('ttk.CheckButton','remove_outliers')

    directory_path, h5_fname = os.path.split(file_path)
    #make directory for processed data from this run, if it doesn't already exist
    directory_path = os.path.join(directory_path, os.path.splitext(h5_fname)[0])
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    #make directory for this export
    settings_name = os.path.splitext(os.path.basename(settings_path))[0]
    directory_path = os.path.join(directory_path, os.path.splitext(h5_fname)[0]+settings_name)
    if  os.path.isdir(directory_path):
        sys.exit("File has already been processed, must be overwritten manually.")
        # if not tkinter.messagebox.askyesno(title='overwrite files?', message='Export already exists, overwrite files?'):
        #     return
    else:
        os.mkdir(directory_path)
    print('Processing run: %s'%h5_fname)
    with open(os.path.join(directory_path, 'config.gui_ini'), 'w') as configfile:
        config.write(configfile)
    #save git version info and modified files
    with open(os.path.join(directory_path, 'git_info.txt'), 'w') as f:
        subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], stdout=f)
        subprocess.run(['git', 'status'], stdout=f)
        subprocess.run(['git', 'diff'], stdout=f)
    #copy channel mapping files
    shutil.copy(flat_lookup_path, directory_path)
    #save event timestamps array
    timestamps = data.get_timestamps_array()
    np.save(os.path.join(directory_path, 'timestamps.npy'), timestamps)
    #save all the other properties
    max_veto_counts, dxy, dz, counts, angles, pads_railed_list = data.get_histogram_arrays()
    np.save(os.path.join(directory_path, 'counts.npy'), counts)
    np.save(os.path.join(directory_path, 'dxy.npy'), dxy)
    np.save(os.path.join(directory_path, 'dt.npy'), dz/data.zscale)
    np.save(os.path.join(directory_path, 'angles.npy'), angles)
    np.save(os.path.join(directory_path, 'veto.npy'), max_veto_counts)
    with open(os.path.join(directory_path, 'pads_railed.csv'), 'w', newline='') as f:
        #TODO: fix railed pads feature so it works with background subtraction turned on
        writer = csv.writer(f)
        writer.writerows(pads_railed_list)
    # max_veto_counts, dxys, dts, counts = max_veto_counts, dxy, dz/data.zscale, counts
    #do zscale dependent calcuations of range and angle


def load_settings_file(self, settings_path):
    config = configparser.ConfigParser()
    file_path = settings_path
    config.read(file_path)
    length_ic_threshold = config.get('ttk.Entry','length_ic_threshold')
    energy_ic_threshold = config.get('ttk.Entry','energy_ic_threshold')
    view_threshold = config.get('ttk.Entry','view_threshold')
    include_cobos = config.get('ttk.Entry','include_cobos')
    include_asads = config.get('ttk.Entry','include_asads')
    include_pads = config.get('ttk.Entry','include_pads')
    veto_threshold = config.get('ttk.Entry','veto_threshold')
    range_min = config.get('ttk.Entry','range_min')
    range_max = config.get('ttk.Entry','range_max')
    min_ic = config.get('ttk.Entry','min_ic')
    max_ic = config.get('ttk.Entry','max_ic')
    angle_min = config.get('ttk.Entry','angle_min')
    angle_max = config.get('ttk.Entry','angle_max')
    background_bin_start = config.get('ttk.Entry','background_bin_start')
    background_bin_stop = config.get('ttk.Entry','background_bin_stop')
    zscale = config.get('ttk.Entry','zscale')
    near_peak_window_width = config.get('ttk.Entry','near_peak_window_width')
    peak_first_allowed_bin = config.get('ttk.Entry','peak_first_allowed_bin')
    peak_last_allowed_bin = config.get('ttk.Entry','peak_last_allowed_bin')
    peak_mode = config.get('ttk.OptionMenu','peak_mode')
    background_mode = config.get('ttk.OptionMenu','background_mode')
    remove_outliers = config.get('ttk.CheckButton','remove_outliers')


    for entry_name in config['ttk.Entry']:
        entry = self.settings_entry_map[entry_name]
        entry.delete(0, tk.END)
        entry.insert(0, config['ttk.Entry'][entry_name])

    for menu_name in config['ttk.OptionMenu']:
        var_to_update = self.settings_optionmenu_map[menu_name]
        var_to_update.set(config['ttk.OptionMenu'][menu_name])

    
    for checkbox_name in config['ttk.CheckButton']:
        var_to_update = self.settings_checkbutton_map[checkbox_name]
        var_to_update.set(config['ttk.CheckButton'][checkbox_name])

    #apply settings to raw data object
    self.entry_changed(None)
    self.check_state_changed()

def process_run(self):
    directory_path, h5_fname = os.path.split(self.data.file_path)
    #make directory for processed data from this run, if it doesn't already exist
    directory_path = os.path.join(directory_path, os.path.splitext(h5_fname)[0])
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    #make directory for this export
    settings_name = os.path.splitext(os.path.basename(self.settings_file_entry.get()))[0]
    directory_path = os.path.join(directory_path, os.path.splitext(h5_fname)[0]+settings_name)
    if  os.path.isdir(directory_path):
        if not tkinter.messagebox.askyesno(title='overwrite files?', message='Export already exists, overwrite files?'):
            return
    else:
        os.mkdir(directory_path)
    print('Processing run: %s'%h5_fname)
    self.save_settings_file(os.path.join(directory_path, 'config.gui_ini'))
    #save git version info and modified files
    with open(os.path.join(directory_path, 'git_info.txt'), 'w') as f:
        subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], stdout=f)
        subprocess.run(['git', 'status'], stdout=f)
        subprocess.run(['git', 'diff'], stdout=f)
    #copy channel mapping files
    shutil.copy(self.data.flat_lookup_file_path, directory_path)
    #save event timestamps array
    self.timestamps = self.data.get_timestamps_array()
    np.save(os.path.join(directory_path, 'timestamps.npy'), self.timestamps)
    #save all the other properties
    max_veto_counts, dxy, dz, counts, angles, pads_railed_list = self.data.get_histogram_arrays()
    np.save(os.path.join(directory_path, 'counts.npy'), counts)
    np.save(os.path.join(directory_path, 'dxy.npy'), dxy)
    np.save(os.path.join(directory_path, 'dt.npy'), dz/self.data.zscale)
    np.save(os.path.join(directory_path, 'angles.npy'), angles)
    np.save(os.path.join(directory_path, 'veto.npy'), max_veto_counts)
    with open(os.path.join(directory_path, 'pads_railed.csv'), 'w', newline='') as f:
        #TODO: fix railed pads feature so it works with background subtraction turned on
        writer = csv.writer(f)
        writer.writerows(pads_railed_list)
    self.max_veto_counts, self.dxys, self.dts, self.counts = max_veto_counts, dxy, dz/self.data.zscale, counts
    #do zscale dependent calcuations of range and angle
    self.entry_changed(None)