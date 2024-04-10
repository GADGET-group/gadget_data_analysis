import datetime
import random
import os
import tkinter.filedialog
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
from matplotlib.path import Path
from GadgetRunH5 import GadgetRunH5
import numpy as np
from tqdm import tqdm
import gadget_widgets
from prev_cut_select_window import PrevCutSelectWindow
from tkinter import filedialog
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision import transforms, models, datasets
import torchvision
import glob
import math
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import InterpolationMode
from collections import defaultdict, Counter
from PIL import Image
from torch.autograd import Variable
from scipy import stats

class CNN_Frame(ttk.Frame):
    def __init__(self, parent, run_data:GadgetRunH5):
        super().__init__(parent)
        self.run_data = run_data
        #show background image
        self.background_image = gadget_widgets.get_background_image()
        self.background = ttk.Label(self, image=self.background_image)
        self.background.place(relx=0.5, rely=0.5, anchor='center')

        #plot settings
        self.cut_tools_frame = ttk.LabelFrame(self, text='Select Data')
        self.cut_tools_frame.grid(row=2)
        #TODO: implement manual cut, and project to axis
        self.manual_cut_button = ttk.Button(self.cut_tools_frame, text='Manual Cut Selection')
        self.manual_cut_button.grid(row=0, column=0)
        self.from_file_cut_button = ttk.Button(self.cut_tools_frame,
                                               text='Polygon from File',
                                               command=self.cut_from_file)
        self.from_file_cut_button.grid(row=0, column=1)
        self.prev_cut_button = ttk.Button(self.cut_tools_frame, 
                                          text='Previous Cuts',
                                          command=self.prev_cut)
        self.prev_cut_button.grid(row=1, column=0, columnspan=2)

        self.model_select_frame = ttk.LabelFrame(self, text='Select & Deploy')
        self.model_select_frame.grid(row=4)
        self.model_select_button = ttk.Button(self.model_select_frame, 
                                              text='Select Trained CNN Model(s)', 
                                              command=self.select_model)
        self.model_select_button.grid(row=1, column=0)

        self.deploy_button = ttk.Button(self.model_select_frame, 
                                              text='Deploy Model(s)', 
                                              command=self.predict)
        self.deploy_button.grid(row=2, column=0)


    def plot_spectrum(self, fig_name='RvE',clear=True, show=True):
        num_range_bins = int(self.range_bins_entry.get())
        num_energy_bins = int(self.energy_bins_entry.get())
        
        plt.figure(fig_name, clear=clear)
        plt.xlabel('Energy (MeV)', fontdict={'fontsize': 20})
        plt.ylabel('Range (mm)', fontdict={'fontsize': 20})
        plt.title(f'Range vs Energy \n Energy Bins = {num_energy_bins} | Range Bins = {num_range_bins}', fontdict={'fontsize': 20})
        tot_energy_temp = np.concatenate(([0], self.run_data.total_energy_MeV))
        len_list_temp = np.concatenate(([0], self.run_data.len_list))
        if self.log_scale_var.get():
            norm = colors.LogNorm()
        else:
            norm = colors.Normalize()
        plt.hist2d(tot_energy_temp, len_list_temp, (num_energy_bins, num_range_bins), 
                   cmap=plt.cm.jet, norm=norm)
        plt.colorbar()
        plt.gca().set_facecolor('darkblue')
        if show:
            plt.show(block=False)

    def show_event(self): #TODO: add "show annotation" checkbox
        #only draw plot if it's not already open
        if not plt.fignum_exists('RvE'):
            self.plot_spectrum()
        else:
            plt.figure('RvE')  # switch focus back to RvE plot
        event_num = int(self.event_num_entry.get())
        event_index = self.run_data.get_index(event_num)
        plt.plot(self.run_data.total_energy_MeV[event_index], self.run_data.len_list[event_index], 'ro', picker=5) 
        plt.annotate(f"Evt#: {event_num}", (self.run_data.total_energy_MeV[event_index], 
                    self.run_data.len_list[event_index]), textcoords="offset points", xytext=(-15,7),
                    ha='center', fontsize=10, color='black',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="black"))
        plt.show(block=False)
        
    def prev_cut(self):
        PrevCutSelectWindow(self, self.run_data)

    def save_cut_files(self, points):
        '''
        points: verticies specifying the cut region
        '''
        now = datetime.datetime.now()
        rand_num = str(random.randrange(0,1000000,1))
        cut_name = rand_num+now.strftime("CUT_Date_%m_%d_%Y")
        event_images_path = os.path.join(self.run_data.folder_path, cut_name)

        #save an image for future cut selection
        self.plot_spectrum(fig_name=cut_name)
        ax = plt.gca()
        #add once point and use this to close the path
        #actual value of final vertex is ignored for CLOSEPOLY code
        points_list = list(points)
        points_list.append([0,0])
        codes = [Path.LINETO]*len(points_list)
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        path = Path(points_list, codes)
        
        to_draw = patches.PathPatch(path, fill=False, color='red')
        ax.add_patch(to_draw)
        plt.savefig(os.path.join(self.run_data.folder_path, cut_name+'.jpg'))
        plt.close()

        #save images of the selected events
        os.makedirs(event_images_path)
        selected_indices = self.run_data.get_RvE_cut_indexes(points)
        for index in tqdm(selected_indices):
            image_name = f"run{self.run_data.run_num}_image_{index}.jpg"
            image_path = os.path.join(event_images_path, image_name)
            self.run_data.save_image(index, save_path=image_path)
        #save the cut parameters used
        np.savetxt(os.path.join(event_images_path, 'cut_used.txt'), points)


    def cut_from_file(self):
        '''
        Input files should have an energy (MeV) followed by range (mm) on each line,
        with the values seperated by a space. 
        '''
        fname = tkinter.filedialog.askopenfile(initialdir=os.getcwd())
        points = np.loadtxt(fname)
        self.save_cut_files(points)

    def select_model(self):
        '''
        Selct model(s)
        All models are trained
        Hold Ctrl to select multiple models and they will be deployed as an ensemble 
        '''
        mypath = f"/mnt/analysis/e21072/models"
        self.models = list(filedialog.askopenfilenames(initialdir=mypath, title="Select a Model"))

    def predict(model_paths): 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Simple Identity class that let's input pass without changes
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()

            def forward(self, x):
                return x
        
        class ImageFolderWithPaths(datasets.ImageFolder):
            def __getitem__(self, index):
                original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
                path = self.imgs[index][0]
                tuple_with_path = (original_tuple + (path,))
                return tuple_with_path

        class CropImage(object):
            def __init__(self, top, bottom, left, right):
                self.top = top
                self.bottom = bottom
                self.left = left
                self.right = right

            def __call__(self, img):
                return img.crop((self.left, self.top, img.width - self.right, img.height - self.bottom))


        # Define transformation
        transform = transforms.Compose([ 
            transforms.ToTensor()
        ])

        def load_model(model_path, device, num_classes):    
            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.avgpool = nn.Identity()
            model.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=True), 
                nn.ReLU(inplace=True), 
                nn.Linear(4096, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes, bias=True)
                )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            return model

        def load_model(model_path, device, num_classes):    
            if not os.path.isfile(model_path):
                print(f"The path {model_path} does not point to a valid file. Skipping this model...")
                return None

            model = models.vgg16(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False
            model.avgpool = nn.Identity()
            model.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=True), 
                nn.ReLU(inplace=True), 
                nn.Linear(4096, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes, bias=True)
            )

            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                return None

            model = model.to(device)
            model.eval()
            return model

        def predict_image_class(image_path, models, device):
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            outputs = [model(image) for model in models]
            avg_output = torch.stack(outputs).mean(0)
            _, predicted_class = torch.max(avg_output, 1)
            return predicted_class.item()

        def predict_directory(directory_path, model_paths, num_classes):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            models = [load_model(model_path, device, num_classes) for model_path in model_paths]
            models = [model for model in models if model is not None]  # Remove any None values

            if not models:
                print("No valid models could be loaded.")
                return {}

            image_paths = glob.glob(os.path.join(directory_path, '*.png'))  # Get all files in the directory
            class_images = defaultdict(list)  # Dictionary where the keys are class indices and the values are lists of image paths
            for image_path in tqdm(image_paths):
                predicted_class = predict_image_class(image_path, models, device)
                class_images[predicted_class].append(image_path)  # Add the image path to the correct class

            # Save the image paths for each class in separate files
            for class_index, image_paths in class_images.items():
                file_path = os.path.join(directory_path, f'class_{class_index}_images.txt')
                if os.path.exists(file_path):
                    print(f"The file {file_path} already exists.")
                    response = input("Do you want to continue and overwrite it? (yes/no): ")
                    if response.lower() != 'yes':
                        print("Skipping this file.")
                        continue
                with open(file_path, 'w') as f:
                    for path in image_paths:
                        f.write(path + '\n')

            return class_images


        
        # paths to your trained models
        # model_paths = [f'./Models/Model_ReduceClasses_Ensemble{i}.pth' for i in [11]]

        # num_classes in your dataset
        num_classes = 3

        predictions = predict_directory(dir_select, model_paths, num_classes)

        print(f"\nPredctions Complete for {model_paths}")

        # You can now print or otherwise use the predictions...
        #for image_path, predicted_class in predictions.items():
        #	print(f"Image: {image_path}, Predicted class: {predicted_class}")