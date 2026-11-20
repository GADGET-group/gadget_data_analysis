from tkinter import ttk
import tkinter as tk
import tkinter.messagebox

from gadget_analysis.main_gui import gadget_widgets, GadgetRunH5

class RunSelectFrame(ttk.Frame):
    '''
    Pick an experiment, a TPC processing config and a GET run; process the run with that config if needed and load
    it. All processing settings (zscale, channel map, baseline subtraction, thresholds, pad gain map) come from the
    config file.
    '''
    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.main_gui = main_gui

        self.background_image = gadget_widgets.get_background_image()
        self.background = tk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')

        self.load_run_frame = ttk.LabelFrame(self, text='select run')
        self.load_run_frame.grid(row=0)

        ttk.Label(self.load_run_frame, text='Experiment:').grid(row=0, column=0, sticky=tk.E)
        experiments = GadgetRunH5.list_experiments()
        self.experiment_combobox = ttk.Combobox(self.load_run_frame, values=experiments)
        if experiments:
            self.experiment_combobox.set(experiments[0])
        self.experiment_combobox.grid(row=0, column=1)
        self.experiment_combobox.bind('<<ComboboxSelected>>', self.experiment_changed)
        self.experiment_combobox.bind('<Return>', self.experiment_changed)
        self.experiment_combobox.bind('<FocusOut>', self.experiment_changed)

        ttk.Label(self.load_run_frame, text='TPC config:').grid(row=1, column=0, sticky=tk.E)
        self.config_combobox = ttk.Combobox(self.load_run_frame, width=30)
        self.config_combobox.grid(row=1, column=1)

        ttk.Label(self.load_run_frame, text='GET run #:').grid(row=2, column=0, sticky=tk.E)
        self.run_number_entry = ttk.Entry(self.load_run_frame)
        self.run_number_entry.grid(row=2, column=1)

        self.apply_veto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.load_run_frame, text="apply the runs module's veto",
                        variable=self.apply_veto_var).grid(row=3, column=0, columnspan=2)

        ttk.Button(self.load_run_frame, text='Process Run', command=self.process_button_clicked).grid(row=4, column=0)
        ttk.Button(self.load_run_frame, text='Load Run', command=self.load_button_clicked).grid(row=4, column=1)

        self.status_label = ttk.Label(self.load_run_frame, text='no run loaded', wraplength=400)
        self.status_label.grid(row=5, column=0, columnspan=2)

        self._configs_for = None
        self.experiment_changed()

        #member variable which will hold run data, once loaded: a GadgetRunH5 object
        self.run_data = None

    def experiment_changed(self, event=None):
        '''Refill the config list for the selected experiment and select its DEFAULT_TPC_CONFIG.'''
        experiment = self.experiment_combobox.get().strip()
        if experiment == self._configs_for:
            return
        self._configs_for = experiment
        configs = GadgetRunH5.list_configs(experiment or None)
        self.config_combobox['values'] = configs
        default = None
        if experiment:
            try:
                default = GadgetRunH5.get_default_config(experiment)
            except Exception as e:   #a runs module that fails to import: leave the choice to the user
                print(f'could not read the default config of {experiment}: {e!r}')
        if default:
            self.config_combobox.set(default)
        elif self.config_combobox.get() not in configs:
            self.config_combobox.set('')

    def get_selection(self):
        '''(experiment, config, run) from the widgets; raises ValueError when one is missing.'''
        experiment = self.experiment_combobox.get().strip()
        config = self.config_combobox.get().strip()
        run_text = self.run_number_entry.get().strip()
        if not experiment:
            raise ValueError('select an experiment')
        if not config:
            raise ValueError('select a TPC config')
        if not run_text:
            raise ValueError('enter a GET run number')
        return experiment, config, int(run_text)

    def set_status(self, text):
        self.status_label['text'] = text
        self.update_idletasks()

    def process(self, experiment, config, run):
        self.set_status(f'processing GET run {run} with {config} (progress in the terminal)...')
        GadgetRunH5.process_run(experiment, run, config)
        self.set_status(f'GET run {run} processed with {config}')

    def process_button_clicked(self):
        experiment, config, run = self.get_selection()
        if GadgetRunH5.is_processed(experiment, run, config):
            tk.messagebox.showinfo('Already processed', f'GET run {run} of {experiment} is already processed with {config}.')
            return
        self.process(experiment, config, run)

    def load_button_clicked(self):
        experiment, config, run = self.get_selection()
        if not GadgetRunH5.is_processed(experiment, run, config):
            if not tk.messagebox.askyesno('Run not processed',
                    f'GET run {run} of {experiment} has not been processed with {config}. Process it now? '
                    '(parallel workers; can take several minutes, progress in the terminal)'):
                return
            self.process(experiment, config, run)
        self.set_status(f'loading GET run {run} ({config})...')
        try:
            self.run_data = GadgetRunH5.GadgetRunH5(experiment, run, config, apply_veto=self.apply_veto_var.get())
        except Exception:
            self.set_status('loading failed')
            raise
        rd = self.run_data
        self.set_status(f'{experiment} GET run {run} ({config}): {len(rd.good_events)} of {rd.num_events} events selected '
                        f'({rd.num_without_track} without a track, {rd.num_vetoed} vetoed)')
        self.main_gui.new_run_loaded()
