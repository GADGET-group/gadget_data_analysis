from tkinter import ttk
import gadget_widgets
import GadgetRunH5
import pandas as pd
import time
import subprocess
import numpy as np

import tkinter as tk
import tkinter.filedialog
import h5py
import tqdm
import os


class SimFrame(ttk.Frame):
    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.main_gui = main_gui
        
        self.background_image = gadget_widgets.get_background_image()
        self.background = tk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')
        
        
        # Simulation Management Frame
        self.manage_sims_frame = ttk.LabelFrame(self, text='Manage Simulations')
        self.manage_sims_frame.grid(row=0)
        
        ttk.Label(self.manage_sims_frame, text="Batch #: s").grid(row=0, column=0)
        self.sim_batch_number_entry = ttk.Entry(self.manage_sims_frame)
        self.sim_batch_number_entry.grid(row=0, column=1)
        self.sim_batch_number_entry.insert(0, "0000")
        
        self.start_sim_button = ttk.Button(self.manage_sims_frame, text='Process Queue', command=self.sim_start_button_clicked)
        self.start_sim_button.grid(row=1, column=0)
        
        self.reset_batch_button = ttk.Button(self.manage_sims_frame, text='Reset Batch', command=self.sim_reset_batch_clicked)
        self.reset_batch_button.grid(row=1, column=1)
        
        self.view_batch_composition_button = ttk.Button(self.manage_sims_frame, text='View Batch', command=self.sim_batch_view)
        self.view_batch_composition_button.grid(row=1,column=2)
        
        ttk.Label(self.manage_sims_frame, text="Select simulation batch by inputting 's####' as the run number").grid(row=2, column=0, columnspan=3)
        
        # Simulation Queue Parameters Frame
        self.sim_queue_params_frame = ttk.LabelFrame(self, text='Queue Parameters')
        self.sim_queue_params_frame.grid(row=1)
        current_row=0
        
        self.create_files_button = ttk.Button(self.sim_queue_params_frame, text='Add to Queue',
                                               command=self.sim_add_to_queue)
        self.create_files_button.grid(row=current_row, column=0)
        
        self.clear_queue_button = ttk.Button(self.sim_queue_params_frame, text='Clear Queue',
                                              command=self.sim_clear_queue)
        self.clear_queue_button.grid(row=current_row, column=1)
        
        self.view_queue_button = ttk.Button(self.sim_queue_params_frame, text='View Queue',
                                            command=self.sim_view_queue)
        self.view_queue_button.grid(row=current_row, column=2)
        
        # Optional Simulation Parameters selection in column 3
        self.default_params_df = pd.read_csv('Simulations/.input/defaultParams.csv') #TODO: non-hardcoded path
        self.sim_queue_optional_params_entries = {}
        self.optional_param_list = self.default_params_df['Parameter'].values
        
        ttk.Label(self.sim_queue_params_frame, text='Optional Parameters:'
                  ).grid(row=current_row, column=3, sticky=tk.E)
        self.sim_queue_optional_params_listbox = tk.Listbox(self.sim_queue_params_frame, selectmode=tk.MULTIPLE, height=4)
        for param in self.optional_param_list:
            self.sim_queue_optional_params_listbox.insert(tk.END, param)  
        self.sim_queue_optional_params_listbox.grid(row=current_row+1, column=3, sticky=tk.NS, rowspan=4)
        self.sim_queue_optional_params_listbox.bind('<<ListboxSelect>>', self.sim_refresh_optional_params)
        
        self.sim_queue_optional_params_scrollbar = ttk.Scrollbar(self.sim_queue_params_frame,
                                                                 orient=tk.VERTICAL,
                                                                 command=self.sim_queue_optional_params_listbox.yview)
        self.sim_queue_optional_params_listbox['yscrollcommand'] = self.sim_queue_optional_params_scrollbar.set
        self.sim_queue_optional_params_scrollbar.grid(row=current_row+1, column=4, sticky=tk.NS, rowspan=4)
        
        # Simulation Queue Parameters
        current_row += 1
        
        tk.Label(self.sim_queue_params_frame, text="Event Type:"
                 ).grid(row=current_row, column=0, sticky=tk.E)
        self.sim_queue_event_type_entry = ttk.Entry(self.sim_queue_params_frame) 
        self.sim_queue_event_type_entry.grid(row=current_row, column=1)
        self.sim_queue_event_type_entry.insert(0, "1200p 500a")
        self.sim_queue_event_b2b_toggle = ttk.Checkbutton(self.sim_queue_params_frame, text=' b2b', variable=tk.BooleanVar())
        self.sim_queue_event_b2b_toggle.grid(row=current_row, column=2)
        current_row +=1

        ttk.Label(self.sim_queue_params_frame, text='Event Quantity:'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.sim_queue_event_quantity_entry = ttk.Entry(self.sim_queue_params_frame)
        self.sim_queue_event_quantity_entry.insert(0,100)
        self.sim_queue_event_quantity_entry.grid(row=current_row, column=1)
        current_row += 1
        
        ttk.Label(self.sim_queue_params_frame, text='Split:'
                  ).grid(row=current_row, column=0, sticky=tk.E)
        self.sim_queue_split_entry = ttk.Entry(self.sim_queue_params_frame)
        self.sim_queue_split_entry.insert(0, 1)
        self.sim_queue_split_entry.grid(row=current_row, column=1)
        current_row += 1
        
        self.sim_queue_optional_params_row = current_row
    
    def sim_start_button_clicked(self):
        # file NA values with defaults
        sim_queue = pd.read_csv('Simulations/parameters.csv')
        for column in sim_queue.columns:
            if column in self.optional_param_list:
                sim_queue[column] = sim_queue[column].fillna(self.default_params_df[self.default_params_df['Parameter'] == column]['Value'].values[0])
        sim_queue.to_csv('Simulations/parameters.csv', index=False)
        
        # run the simulation
        subprocess.run([f'Simulations/run_sim.sh', '-m4'])
        
        # merge the simulations into a batch
        self.merge_sims_to_batch()
        
    
    def sim_refresh_optional_params(self, event):
        selected_indices = self.sim_queue_optional_params_listbox.curselection()
        
        # don't reset modified values
        saved_values = {}
        if hasattr(self, 'sim_queue_optional_params_entries'):
            for param, entry in self.sim_queue_optional_params_entries.items():
                saved_values[param] = entry.get()
        
        start_row = self.sim_queue_optional_params_row
        self.sim_queue_optional_params_entries = {} # clear the old entries

        # clear the old entries from the frame
        for widget in self.sim_queue_params_frame.winfo_children():
            if widget.grid_info()['row'] > start_row:
                widget.destroy()
        
        if len(selected_indices) == 0:
            return
        
        optional_params = " ".join([self.optional_param_list[i] for i in selected_indices])
        optional_params = optional_params.split()
        
        current_row = start_row + 1
        for param in optional_params:
            ttk.Label(self.sim_queue_params_frame, text=f'{param}'
                    ).grid(row=current_row, column=0, sticky=tk.E)
            self.sim_queue_optional_params_entries[param] = ttk.Entry(self.sim_queue_params_frame)
            
            if param in saved_values.keys(): # restore the old value if it exists
                self.sim_queue_optional_params_entries[param].insert(0, saved_values[param])
            else: # otherwise, insert the default value
                self.sim_queue_optional_params_entries[param].insert(0, self.default_params_df[self.default_params_df['Parameter'] == param]['Value'].values[0])
            self.sim_queue_optional_params_entries[param].grid(row=current_row, column=1)
            current_row += 1

    def string_to_param(self, param_name:str, input_string:str):
        # parameter types: int, float, string
        if param_name in self.optional_param_list:
            param_type = self.default_params_df[self.default_params_df['Parameter'] == param_name]['Type'].values[0]
        elif param_name in ['E0', 'E1']:
            param_type = 'float'
        elif param_name in ['P0', 'P1']:
            param_type = 'string'
        elif param_name in ['N']:
            param_type = 'int'
        else:
            raise ValueError(f"Parameter {param_name} not identified.")
        input_string = str(input_string)
        if param_type == 'string':
            # no conversions needed
            return input_string
        
        # interpret the input string
        '''
        input string format options:
        - single value: {value}
        
        - normal distribution: {mean}+{stddev}
            - stddev can be a percentage of the mean: {mean}+{stddev}%
        
        - uniform distribution: {min}:{max}
        - triangular distribution: {min}:{mode}:{max}
        
        todo:
        - uniform distribution with % variation from center
        '''
        
        if '+' in input_string: # normal distribution
            # get the mean and stddev
            mean, stddev = input_string.split('+')
            mean = float(mean)
            if stddev[-1] == '%':
                stddev = mean * float(stddev[:-1]) / 100
            else:
                stddev = float(stddev)
            
            # generate the random value    
            if param_type == 'int':
                return int(np.random.normal(mean, stddev))
            else:
                return np.random.normal(mean, stddev)
            
        elif ':' in input_string:
            values = input_string.split(':')
            if len(values) == 2:
                min_val, max_val = float(values[0]), float(values[1])
                if param_type == 'int':
                    return np.random.randint(min_val, max_val)
                else:
                    return np.random.uniform(min_val, max_val)
            elif len(values) == 3:
                min_val, mode_val, max_val = float(values[0]), float(values[1]), float(values[2])
                if param_type == 'int':
                    return int(np.random.triangular(min_val, mode_val, max_val))
                else:
                    return np.random.triangular(min_val, mode_val, max_val)
        else:
            if param_type == 'int':
                return int(input_string)
            else:
                return float(input_string)
    
    def validate_params(self, append_dict):
        for param, value in append_dict.items():
            if param in self.optional_param_list:
                param_type = self.default_params_df[self.default_params_df['Parameter'] == param]['Type'].values[0]
                param_min = self.default_params_df[self.default_params_df['Parameter'] == param]['Min'].values[0]
                param_max = self.default_params_df[self.default_params_df['Parameter'] == param]['Max'].values[0]
            elif param in ['E0', 'E1']:
                param_type = 'float'
                param_min = 0
                param_max = np.inf
            elif param in ['P0', 'P1', 'Sim']:
                param_type = 'string'
                param_min = ''
                param_max = ''
            elif param in ['N', 'Status']:
                param_type = 'int'
                param_min = 0
                param_max = np.inf
            else:
                print(f"{param} not identified.")
                return False
            
            # validate matching types
            if param_type == 'string':
                continue # no validation needed
            elif param_type == 'int' and not isinstance(value, int):
                print(f"{param} : {value} is not an integer.")
                return False
            elif param_type == 'float' and not isinstance(value, float):
                print(f"{param} : {value} is not a float.")
                return False
            
            if param_min == '' : # no lower bound
                param_min = -np.inf
            else:
                param_min = float(param_min)
            if param_max == '' : # no upper bound
                param_max = np.inf
            else:
                param_max = float(param_max)
                
            
            # check if the value is within the bounds
            if value < param_min:
                print(f"{param} : {value} is less than the minimum value of {param_min}.")
                return False
            elif value > param_max:
                print(f"{param} : {value} is greater than the maximum value of {param_max}.")
                return False
        return True # no errors found
            
    
    def sim_add_to_queue(self):
        sim_queue = pd.read_csv('Simulations/parameters.csv') #TODO: non-hardcoded path
        
        # get the parameters from the GUI
        event_type = self.sim_queue_event_type_entry.get()
        event_quantity = int(self.sim_queue_event_quantity_entry.get())
        split = int(self.sim_queue_split_entry.get())
        
        if event_quantity < 1:
            raise ValueError("Event quantity must be greater than 0.")
        if split < 1:
            raise ValueError("Split must be greater than 0.")
        
        optional_params = {}
        for param, entry in self.sim_queue_optional_params_entries.items():
            optional_params[param] = entry.get()
        
        # split event type into its components
        event_type = event_type.split()
        if len(event_type) == 1: # single event type
            E0, P0 = event_type[0][:-1], event_type[0][-1]
            E1, P1 = 0, 'a'
        elif len(event_type) == 2: # two event types
            E0, P0 = event_type[0][:-1], event_type[0][-1]
            E1, P1 = event_type[1][:-1], event_type[1][-1]
        elif len(event_type) > 2:
            raise ValueError("3+ particle events are currently not supported.")
        # b2b toggle
        if self.sim_queue_event_b2b_toggle.instate(['selected']):
            P1 = f'-{P1}'
        
        for i in range(split):
            for attempts in range(10): 
                append_dict = {
                    'Sim': time.time(),
                    'E0': self.string_to_param('E0', E0),
                    'P0': P0,
                    'E1': self.string_to_param('E1', E1),
                    'P1': P1,
                    'N': event_quantity//split,
                    'Status': 0
                }
                for param, value in optional_params.items():
                    append_dict[param] = self.string_to_param(param, value)
            
                # validate the parameters
                if self.validate_params(append_dict):
                    sim_queue = sim_queue.append(append_dict, ignore_index=True)
                    break
                elif split == 1: # if split is 1, don't try again
                    print(f"Validation of parameters for single-split failed, not adding to queue.")
                    return # exit the function
                elif attempts == 9:
                    print(f"Failed to validate parameters after 10 attempts. Skipping {i}.")
                    continue
        
        # save the updated queue
        sim_queue.to_csv('Simulations/parameters.csv', index=False)
    
    
    def sim_reset_batch_clicked(self):
        sim_number = GadgetRunH5.run_num_to_str(self.sim_batch_number_entry.get())
        if os.path.exists(f"./Simulations/Batches/run_s{sim_number}"):
            num_components = len(os.listdir(f"./Simulations/Batches/run_s{sim_number}/component_h5"))
            print(f"Confirm deletion of {num_components} existing simulation files in batch {sim_number}?")
            input_str = input("Type 'yes' to confirm: ")
            if input_str == 'yes':
                os.system(f"rm -r ./Simulations/Batches/run_s{sim_number}")
                os.system(f"rm ./Simulations/Batches/run_s{sim_number}.h5")
                print(f"Batch {sim_number} has been reset.")
        else:
            print(f"Batch {sim_number} is empty.")
    
    def sim_batch_view(self):
        sim_number = int(self.sim_batch_number_entry.get())
        if os.path.exists(f"./Simulations/Batches/run_s{sim_number}"):
            batch_composition = os.listdir(f"./Simulations/Batches/run_s{sim_number}/component_h5")
            print(f"Batch {sim_number} contains {len(batch_composition)} components.")
            print(batch_composition)
        else:
            print(f"Batch {sim_number} is empty.")
    
    def sim_clear_queue(self):
        sim_queue = pd.read_csv('Simulations/parameters.csv')
        
        if len(sim_queue) < 1:
            print("Simulation queue is already empty.")
            return
        else:
            print(f"Confirm deletion of {len(sim_queue)} queued simulations?")
            
            input_str = input("Type 'yes' to confirm: ")
            if input_str == 'yes':
                sim_queue = pd.DataFrame(columns=['Sim', 'E0', 'P0', 'E1', 'P1', 'N', 'Status'])
                sim_queue.to_csv('Simulations/parameters.csv', index=False)
                print("Simulation queue has been cleared.")
    
    def sim_view_queue(self):
        sim_queue = pd.read_csv('Simulations/parameters.csv')
        print(sim_queue)
    

    def merge_sims_to_batch(self):
        Simulation_path = "./Simulations/Batches/"
        Queue_path = "Simulations/Output/"
        
        batch_number = self.sim_batch_number_entry.get()
        batch_name = f"run_s{batch_number}"
        
        print(f"Merging simulations to batch {batch_name}")
        
        # check if the batch already exists
        if not os.path.exists(f"{Simulation_path}{batch_name}"):
            os.makedirs(f"{Simulation_path}{batch_name}")
            os.makedirs(f"{Simulation_path}{batch_name}/component_h5")
        
        #subprocess.run(f"cp {Simulation_path}parameters.csv {Simulation_path}{batch_name}/component_h5/param-{time.time()}.csv", shell=True, check=True)
        
        # move all component files to batch
        sim_files = os.listdir(Queue_path)
        for sim_file in sim_files:
            if sim_file.endswith(".h5"):
                os.rename(f"{Queue_path}{sim_file}", f"{Simulation_path}{batch_name}/component_h5/{sim_file}")
        
        # create the batch h5 file
        if os.path.exists(f"{Simulation_path}{batch_name}.h5"):
            os.remove(f"{Simulation_path}{batch_name}.h5")
        
        if len(os.listdir(f"{Simulation_path}{batch_name}/component_h5")) < 1:
            print("No component files found for batch.")
            return
        
        component_h5_list = [filename for filename in os.listdir(f"{Simulation_path}{batch_name}/component_h5") if filename.endswith('.h5')]
        component_h5_list.sort()
        
        # copy first file as template
        template_h5 = component_h5_list.pop(0)
        os.system(f"cp {Simulation_path}{batch_name}/component_h5/{template_h5} {Simulation_path}{batch_name}.h5")
        
        for component_h5 in component_h5_list:
            batch_h5 = h5py.File(f"{Simulation_path}{batch_name}.h5", 'a')
            component_h5 = h5py.File(f"{Simulation_path}{batch_name}/component_h5/{component_h5}", 'r')
            
            # get the event number of the last event in the batch
            last_event_number = int(batch_h5['meta/meta'][2]) + 1
            
            # merge get data
            for key in component_h5['get'].keys():
                if 'data' in key:
                    try:
                        event_number = int(key.split('_')[0][3:])
                        batch_h5.create_dataset(f"get/evt{event_number + last_event_number}_header", data=component_h5[f"get/{key}"], dtype='float64')
                        batch_h5.create_dataset(f"get/evt{event_number + last_event_number}_data", data=component_h5[f"get/{key}"], dtype='int16')
                        batch_h5[f'get/evt{event_number + last_event_number}_header'][0] = event_number + last_event_number # update the event number in the header
                    except IndexError:
                        print(f"Error in {component_h5} - {key}")
            meta_data = batch_h5['meta/meta'] 
            
            # merge clouds data
            for key in component_h5['clouds'].keys():
                event_number = int(key.split('_')[0].split('t')[1]) # evt#_cloud
                batch_h5.create_dataset(f"clouds/evt{event_number + last_event_number}_cloud", data=component_h5[f"clouds/{key}"], dtype='float64')
                meta_data[2] = max(meta_data[2], event_number + last_event_number)
            
            # re-write the meta data
            batch_h5['meta'].pop('meta')
            batch_h5.create_dataset('meta/meta', data=meta_data, dtype='float64')
            
            # close the files
            component_h5.close()
            batch_h5.close()
